"""
Camera -> BEV models used to get a BEV feature map and then a segmentation output from it.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from bev.data.nuscenes_dataset import CameraDataBatch
from bev.raster import BEVGridSpec

class ResNetBackbone(nn.Module):
    """
    Frozen-weight ResNet backbone that will provide the image-space feature map.
    
    Using the ResNet layers, we add, on-top, upsampling layers and add-skip connections (FPN-style), and output the last layer (stride-4 of the original input with 128 features).
    """
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.requires_grad_(False)
        self._freeze_bn()
        self.stem = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer5conv = nn.Conv2d(512, 256, (1,1)) # applied on upsampled layer 4 before adding to layer 3 to smooth and get layer 5
        self.layer5smooth = nn.Conv2d(256, 256, (3,3), padding=(1,1)) # 3x3 smoothing to get layer 5
        self.layer6conv = nn.Conv2d(256, 128, (1,1)) # applied on upsampled layer 5 before adding to layer 2 to smooth and get layer 6
        self.layer6smooth = nn.Conv2d(128, 128, (3,3), padding=(1,1)) # 3x3 smoothing to get layer 6
        self.layer7conv = nn.Conv2d(64, 128, (1,1)) # applied on layer 1 before adding to upsampled layer 6 to smooth and get layer 7 = output
        self.layer7smooth = nn.Conv2d(128, 128, (3,3), padding=(1,1)) # 3x3 smoothing to get layer 7
    
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._freeze_bn()
        return self
    
    def forward(self, x: torch.Tensor): # in: (B, 3, H, W), out: (B,128,H/4,W/4)
        stem_out = self.stem(x)
        layer1_out = self.resnet.layer1(stem_out)
        layer2_out = self.resnet.layer2(layer1_out)
        layer3_out = self.resnet.layer3(layer2_out)
        layer4_out = self.resnet.layer4(layer3_out)
        layer5_out = self.layer5smooth(layer3_out+self.layer5conv(F.interpolate(layer4_out, size=layer3_out.shape[-2:], mode='bilinear', align_corners=False)))
        layer6_out = self.layer6smooth(layer2_out+self.layer6conv(F.interpolate(layer5_out, size=layer2_out.shape[-2:], mode='bilinear', align_corners=False)))
        layer7_out = self.layer7smooth(self.layer7conv(layer1_out)+F.interpolate(layer6_out, size=layer1_out.shape[-2:], mode='bilinear', align_corners=False))

        return layer7_out 

class FeatureDepthPredictor(nn.Module):
    """
    An MLP on top of the features computed by the backbone, to predict the depth of each pixel as a distribution over D depth bins.
    """
    def __init__(self, D: int = 45):
        super().__init__()
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(128, 64, (1,1)),
            nn.ReLU(),
            nn.Conv2d(64, D, (1,1)),
        )

    def forward(self, x: torch.Tensor): # (B,128,H/4,W/4) -> (B, D, H/4, W/4)
        return self.depth_predictor(x)

class BEVLift(nn.Module):
    """
    Lifting module that uses the outputs (feature map at stride 4 of the input images) of a CNN backbone (like ImageNet) and composes them in the BEV frame through projection at various BEV heights.

    It doesn't add any new parameters on top of the backbone.
    """
    def __init__(self, backbone: nn.Module, depth_predictor: nn.Module, bev_raster_spec: BEVGridSpec, min_z : float, max_z: float, num_z: int, min_d: float, max_d: float, D: int):
        super().__init__()
        self.backbone = backbone
        self.depth_predictor = depth_predictor
        self.bev_raster_spec = bev_raster_spec
        jj, ii, iz = torch.meshgrid([torch.arange(self.bev_raster_spec.ny), torch.arange(self.bev_raster_spec.nx), torch.arange(num_z)]) # (ny, nx, nz), y-first to match the GT raster
        zlin = torch.linspace(min_z, max_z, num_z)
        zs = zlin[iz]
        xs = self.bev_raster_spec.x_min + (ii+0.5)*self.bev_raster_spec.resolution # (ny, nx, nz)
        ys = self.bev_raster_spec.y_min + (jj+0.5)*self.bev_raster_spec.resolution # (ny, nx, nz)
        self.register_buffer("bev_norm_coords", torch.stack([xs, ys, zs, torch.ones_like(xs)])) # (4, ny, nx, nz)
        self.d_res = (max_d-min_d)/(D-1)
        self.D = D
        self.min_d = min_d
        self.max_d = max_d

    def forward(self, x: CameraDataBatch): # out: (B,128,ny,nx)
        backbone_out = self.backbone(x.images.reshape(-1,3,x.images.shape[3],x.images.shape[4])) # (B*N,128,H/4,W/4)
        depth_out = self.depth_predictor(backbone_out) # (B*N,D,H/4,W/4)
        backbone_out = backbone_out.reshape(x.images.shape[0], x.images.shape[1], backbone_out.shape[1], backbone_out.shape[2], backbone_out.shape[3]) # (B,N,128,H/4,W/4)
        depth_out = depth_out.reshape(x.images.shape[0], x.images.shape[1], depth_out.shape[1], depth_out.shape[2], depth_out.shape[3]) # (B,N,D,H/4,W/4)
        depth_probs_out = torch.softmax(depth_out, dim=2) # (B,N,D,H/4,W/4)
        projected_norm_coords = torch.tensordot(x.bev2pixel, self.bev_norm_coords, dims=([3],[0])) # (B,N,3,ny,nx,nz)
        depth_lift = projected_norm_coords[:,:,2,:,:,:] # (B,N,nx,ny,nz)
        projected_coords = projected_norm_coords[:,:,:2,:,:,:] / projected_norm_coords[:,:,2:3,:,:,:] # (B,N,2,nx,ny,nz)
        backbone_coords = (projected_coords/4).to(torch.int32) # (B,N,2,nx,ny,nz), backbone output pixel coords
        ib, inn, ix, iy, iz = torch.meshgrid([torch.arange(x.images.shape[0]), torch.arange(x.images.shape[1]), torch.arange(self.bev_norm_coords.shape[1]), torch.arange(self.bev_norm_coords.shape[2]), torch.arange(self.bev_norm_coords.shape[3])])
        v = backbone_coords[:,:,1,:,:,:]
        u = backbone_coords[:,:,0,:,:,:]
        camera_mask = (depth_lift >= self.min_d) & \
                    (depth_lift <= self.max_d) & \
                    (u >= 0) & \
                    (v >= 0) & \
                    (u < backbone_out.shape[4]) & \
                    (v < backbone_out.shape[3])
        depth_lift = depth_lift.clamp(min=self.min_d, max=self.max_d)
        v = v.clamp(0,backbone_out.shape[3]-1)
        u = u.clamp(0,backbone_out.shape[4]-1)
        lo_depth = torch.floor((depth_lift-self.min_d)/self.d_res).int().clamp(min=0, max=self.D-2)
        hi_depth = lo_depth + 1
        lo_frac = 1-(depth_lift - self.min_d - lo_depth*self.d_res)/self.d_res
        hi_frac = 1-lo_frac
        depth_prob = lo_frac*depth_probs_out[ib, inn, lo_depth, v,u]+hi_frac*depth_probs_out[ib, inn, hi_depth, v, u]
        bev_from_backbone = torch.where(camera_mask.unsqueeze(2), backbone_out[ib, inn, :, v, u].movedim(-1,2)*depth_prob.unsqueeze(2), 0.0) # (B,N,128,nx,ny,nz)
        return (bev_from_backbone.sum(dim=1)/camera_mask.sum(dim=1).unsqueeze(1).clamp(min=1)).sum(dim=4) # mean over cameras, sum over z

class CameraBEVForwardScatter(nn.Module):
    """
    Forward scatter module (LSS-like) that uses the outputs (feature map & depth distribution at stride 4 of the input images) of a CNN backbone (like ImageNet) and scatters them in the BEV frame.

    It doesn't add any new parameters on top of the backbone.
    """
    def __init__(self, backbone: nn.Module, depth_predictor: nn.Module, bev_raster_spec: BEVGridSpec, min_d: float, max_d: float, D: int):
        super().__init__()
        self.backbone = backbone
        self.depth_predictor = depth_predictor
        self.bev_raster_spec = bev_raster_spec
        self.d_res = (max_d-min_d)/(D-1)
        self.D = D
        self.min_d = min_d
        self.max_d = max_d

    def forward(self, x: CameraDataBatch): # out: (B,128,nx,ny)
        backbone_out = self.backbone(x.images.reshape(-1,3,x.images.shape[3],x.images.shape[4])) # (B*N,128,H/4,W/4)
        depth_out = self.depth_predictor(backbone_out) # (B*N,D,H/4,W/4)
        backbone_out = backbone_out.reshape(x.images.shape[0], x.images.shape[1], backbone_out.shape[1], backbone_out.shape[2], backbone_out.shape[3]) # (B,N,128,H/4,W/4)
        depth_out = depth_out.reshape(x.images.shape[0], x.images.shape[1], depth_out.shape[1], depth_out.shape[2], depth_out.shape[3]) # (B,N,D,H/4,W/4)
        depth_probs_out = torch.softmax(depth_out, dim=2) # (B,N,D,H/4,W/4)

        inv_intrinsics = torch.linalg.inv(x.intrinsics).to(backbone_out.device) # (B,N,3,3)
        ego2bev = torch.linalg.inv(x.bev2ego).to(backbone_out.device) # (B,N,4,4)
        uu, vv = torch.meshgrid([torch.arange(backbone_out.shape[3], device=backbone_out.device), torch.arange(backbone_out.shape[4], device=backbone_out.device)])
        uu, vv = 4*uu.float(), 4*vv.float() # image coordinates, (H/4, W/4)

        B = x.images.shape[0]
        N = x.images.shape[1]
        H_4 = backbone_out.shape[3]
        W_4 = backbone_out.shape[4]

        unscaled_frustum_ego = inv_intrinsics.reshape(B,N,1,1,3,3) @ torch.stack([vv, uu, torch.ones_like(uu)], dim=2).reshape(1,1,H_4,W_4,3,1) # (B,N,H/4,W/4,3,1)

        cum_bev_features = torch.zeros((B,backbone_out.shape[2],self.bev_raster_spec.nx,self.bev_raster_spec.ny), dtype=torch.float32, device=backbone_out.device)
        cum_bev_cameras = torch.zeros((B,N,self.bev_raster_spec.nx,self.bev_raster_spec.ny), dtype=torch.int32, device=backbone_out.device)

        ib, inn, ic, ih_4, iw_4 = torch.meshgrid([torch.arange(B, device=backbone_out.device), torch.arange(N, device=backbone_out.device), torch.arange(backbone_out.shape[2], device=backbone_out.device), torch.arange(H_4, device=backbone_out.device), torch.arange(W_4, device=backbone_out.device)])


        for d in range(self.D):
            depth = self.min_d + d*self.d_res
            frustum_bev = (ego2bev.reshape(B,N,1,1,4,4) @ torch.concat([depth*unscaled_frustum_ego, torch.ones((B,N,H_4,W_4,1,1), dtype=torch.float32, device=backbone_out.device)], dim=-2)).reshape(B,N,H_4,W_4,4)
            frustum_bev = (frustum_bev/frustum_bev[:,:,:,:,3:4])[:,:,:,:,:2] # (B,N,H/4,W/4,2)
            frustum_bev_grid = (((frustum_bev - torch.tensor([self.bev_raster_spec.x_min, self.bev_raster_spec.y_min], dtype=torch.float32, device=backbone_out.device).reshape(1,1,1,1,2)) \
                /torch.tensor([self.bev_raster_spec.x_max - self.bev_raster_spec.x_min, self.bev_raster_spec.y_max - self.bev_raster_spec.y_min], dtype=torch.float32, device=backbone_out.device).reshape(1,1,1,1,2)) \
                *torch.tensor([self.bev_raster_spec.nx, self.bev_raster_spec.ny], dtype=torch.float32, device=backbone_out.device).reshape(1,1,1,1,2)).int()
            # (B,N,H/4,W/4,2)
            
            in_grid = (frustum_bev_grid[...,0] >= 0) & (frustum_bev_grid[...,1] >= 0) & (frustum_bev_grid[...,0] < self.bev_raster_spec.nx) & (frustum_bev_grid[...,1] < self.bev_raster_spec.ny)
            # (B,N,H/4,W/4)

            frustum_bev_grid[...,0] = frustum_bev_grid[...,0].clamp(min=0,max=self.bev_raster_spec.nx-1)
            frustum_bev_grid[...,1] = frustum_bev_grid[...,1].clamp(min=0,max=self.bev_raster_spec.ny-1)

            cum_bev_features.index_put_((ib, ic, frustum_bev_grid[:,:,None,:,:,1], frustum_bev_grid[:,:,None,:,:,0]), torch.where(in_grid[:,:,None,:,:], backbone_out*depth_probs_out[:, :, d].unsqueeze(2), 0), accumulate=True)
            cum_bev_cameras.index_put_((ib, inn, frustum_bev_grid[:,:,None,:,:,1], frustum_bev_grid[:,:,None,:,:,0]), torch.where(in_grid[:,:,None,:,:], 1, 0).int(), accumulate=True)
        
        return cum_bev_features/(cum_bev_cameras > 0).sum(dim=1).reshape(B,-1,self.bev_raster_spec.nx,self.bev_raster_spec.ny).clamp(min=1)

class BEVSeg(nn.Module):
    """
    Segmentation head that uses the output of a BEV lifting module (like BEVLift) to produce segmentation results for a set of labels in the BEV grid.
    """
    def __init__(self, bev_raster_labels: tuple[str, ...]):
        super().__init__()
        self.bev_raster_labels = bev_raster_labels
        self.convs = nn.Sequential(
            nn.Conv2d(128, 64, (5,5), padding=(2,2)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3,3), padding=(1,1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(32, len(self.bev_raster_labels), (1,1)),
        )
    
    def forward(self, x: torch.Tensor): # in: (B, 128, nx, ny), out: (B, num_labels, nx, ny)
        return self.convs(x)

class CameraBEVSeg(nn.Module):
    """
    A composition of the ResNet backbone, BEV lifting, and BEV segmentation models to get BEV segments directly from input image data.
    """
    def __init__(self, bev_raster_spec: BEVGridSpec, bev_raster_labels: tuple[str, ...], min_z : float, max_z: float, num_z: int, min_d: float, max_d: float, D: int):
        super().__init__()
        self.bevlift = BEVLift(ResNetBackbone(), FeatureDepthPredictor(D), bev_raster_spec, min_z, max_z, num_z, min_d, max_d, D)
        self.bevseg = BEVSeg(bev_raster_labels)
    
    def forward(self, x: CameraDataBatch):
        return self.bevseg(self.bevlift(x))

class CameraBEVSegScatter(nn.Module):
    """
    A composition of the ResNet backbone, BEV lifting, and BEV segmentation models to get BEV segments directly from input image data.
    """
    def __init__(self, bev_raster_spec: BEVGridSpec, bev_raster_labels: tuple[str, ...], min_d: float, max_d: float, D: int):
        super().__init__()
        self.bevlift = CameraBEVForwardScatter(ResNetBackbone(), FeatureDepthPredictor(D), bev_raster_spec, min_d, max_d, D)
        self.bevseg = BEVSeg(bev_raster_labels)

    def forward(self, x: CameraDataBatch):
        return self.bevseg(self.bevlift(x))

class BEVLiftProjection(nn.Module):
    """
    Projection-only BEV lift (no depth prediction): projects the BEV grid into each
    camera, nearest-samples the stride-4 backbone features, masks out-of-view cells,
    means over cameras and sums over height. The depth-free baseline for BEVLift.

    It doesn't add any new parameters on top of the backbone.
    """
    def __init__(self, backbone: nn.Module, bev_raster_spec: BEVGridSpec, min_z: float, max_z: float, num_z: int):
        super().__init__()
        self.backbone = backbone
        self.bev_raster_spec = bev_raster_spec
        jj, ii, iz = torch.meshgrid([torch.arange(self.bev_raster_spec.ny), torch.arange(self.bev_raster_spec.nx), torch.arange(num_z)]) # (ny, nx, nz), y-first to match the GT raster
        zlin = torch.linspace(min_z, max_z, num_z)
        zs = zlin[iz]
        xs = self.bev_raster_spec.x_min + (ii+0.5)*self.bev_raster_spec.resolution # (ny, nx, nz)
        ys = self.bev_raster_spec.y_min + (jj+0.5)*self.bev_raster_spec.resolution # (ny, nx, nz)
        self.register_buffer("bev_norm_coords", torch.stack([xs, ys, zs, torch.ones_like(xs)])) # (4, ny, nx, nz)

    def forward(self, x: CameraDataBatch): # out: (B,128,ny,nx)
        backbone_out = self.backbone(x.images.reshape(-1,3,x.images.shape[3],x.images.shape[4])) # (B*N,128,H/4,W/4)
        backbone_out = backbone_out.reshape(x.images.shape[0], x.images.shape[1], backbone_out.shape[1], backbone_out.shape[2], backbone_out.shape[3]) # (B,N,128,H/4,W/4)
        projected_norm_coords = torch.tensordot(x.bev2pixel, self.bev_norm_coords, dims=([3],[0])) # (B,N,3,ny,nx,nz)
        projected_coords = projected_norm_coords[:,:,:2,:,:,:] / projected_norm_coords[:,:,2:3,:,:,:] # (B,N,2,ny,nx,nz)
        backbone_coords = (projected_coords/4).to(torch.int32) # (B,N,2,ny,nx,nz), backbone output pixel coords
        ib, inn, ix, iy, iz = torch.meshgrid([torch.arange(x.images.shape[0]), torch.arange(x.images.shape[1]), torch.arange(self.bev_norm_coords.shape[1]), torch.arange(self.bev_norm_coords.shape[2]), torch.arange(self.bev_norm_coords.shape[3])])
        camera_mask = (projected_norm_coords[:,:,2,:,:] >= 0) & \
                    (backbone_coords[:,:,0,:,:,:] >= 0) & \
                    (backbone_coords[:,:,1,:,:,:] >= 0) & \
                    (backbone_coords[:,:,0,:,:,:] < backbone_out.shape[4]) & \
                    (backbone_coords[:,:,1,:,:,:] < backbone_out.shape[3])
        bev_from_backbone = torch.where(camera_mask.unsqueeze(2), backbone_out[ib, inn, :, backbone_coords[:,:,1,:,:,:].clamp(0,backbone_out.shape[3]-1), backbone_coords[:,:,0,:,:,:].clamp(0,backbone_out.shape[4]-1)].movedim(-1,2), 0.0) # (B,N,128,ny,nx,nz)
        return (bev_from_backbone.sum(dim=1)/camera_mask.sum(dim=1).unsqueeze(1).clamp(min=1)).sum(dim=4) # mean over cameras, sum over z

class CameraBEVSegProjection(nn.Module):
    """
    A composition of the ResNet backbone, projection-only BEV lifting, and BEV
    segmentation, to get BEV segments directly from input image data.
    """
    def __init__(self, bev_raster_spec: BEVGridSpec, bev_raster_labels: tuple[str, ...], min_z: float, max_z: float, num_z: int):
        super().__init__()
        self.bevlift = BEVLiftProjection(ResNetBackbone(), bev_raster_spec, min_z, max_z, num_z)
        self.bevseg = BEVSeg(bev_raster_labels)

    def forward(self, x: CameraDataBatch):
        return self.bevseg(self.bevlift(x))

class FocalLoss(nn.Module):
    """
    Focal loss (per-pixel) applied on the respective classes.
    """
    def __init__(self, gamma: float = 2, alphas: tuple[float, ...] = (0.25, 0.5)):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alphas", torch.Tensor(alphas).reshape(1,-1,1,1))

    def forward(self, logits: torch.Tensor, target: torch.Tensor): # (B, num_labels, nx, ny)
        p = torch.sigmoid(logits)
        return (self.alphas*(1-p)**self.gamma*(-F.logsigmoid(logits)*target)+(1-self.alphas)*p**self.gamma*(-F.logsigmoid(-logits)*(1-target))).mean()

class DiceLoss(nn.Module):
    """
    Dice loss (across pixels) applied on the respcetive classes.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor): # (B, num_labels, nx, ny)
        p = torch.sigmoid(logits)
        return (1-(2*(p*target).sum(dim=(2,3))+self.eps)/(p.sum(dim=(2,3))+target.sum(dim=(2,3))+self.eps)).mean()

class SegLoss(nn.Module):
    """
    Weighed focal with Dice loss for the segmentation output vs ground truth.
    """
    def __init__(self, gamma: float = 2, alphas: tuple[float, ...] = (0.25, 0.5), eps: float = 1e-5, lamda: float = 1):
        super().__init__()
        self.focal_loss = FocalLoss(gamma, alphas)
        self.dice_loss = DiceLoss(eps)
        self.lamda = lamda

    def forward(self, logits: torch.Tensor, target: torch.Tensor): # (B, num_labels, nx, ny)
        return self.focal_loss(logits, target)+self.lamda*self.dice_loss(logits, target)