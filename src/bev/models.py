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

class BEVLift(nn.Module):
    """
    Lifting module that uses the outputs (feature map at stride 4 of the input images) of a CNN backbone (like ImageNet) and composes them in the BEV frame through projection at various BEV heights.

    It doesn't add any new parameters on top of the backbone.
    """
    def __init__(self, backbone: nn.Module, bev_raster_spec: BEVGridSpec, min_z : float, max_z: float, num_z: int):
        super().__init__()
        self.backbone = backbone
        self.bev_raster_spec = bev_raster_spec
        ii, jj, iz = torch.meshgrid([torch.arange(self.bev_raster_spec.nx), torch.arange(self.bev_raster_spec.ny), torch.arange(num_z)]) # (nx, ny, nz)
        zlin = torch.linspace(min_z, max_z, num_z)
        zs = zlin[iz]
        xs = self.bev_raster_spec.x_min + (ii+0.5)*self.bev_raster_spec.resolution # (nx, ny, nz)
        ys = self.bev_raster_spec.y_min + (jj+0.5)*self.bev_raster_spec.resolution # (nx, ny, nz)
        self.register_buffer("bev_norm_coords", torch.stack([xs, ys, zs, torch.ones_like(xs)])) # (4, nx, ny, nz)

    def forward(self, x: CameraDataBatch): # out: (B,128,nx,ny)
        backbone_out = self.backbone(x.images.reshape(-1,3,x.images.shape[3],x.images.shape[4])) # (B*N,128,H/4,W/4)
        backbone_out = backbone_out.reshape(x.images.shape[0], x.images.shape[1], backbone_out.shape[1], backbone_out.shape[2], backbone_out.shape[3]) # (B,N,128,H/4,W/4)
        projected_norm_coords = torch.tensordot(x.bev2pixel, self.bev_norm_coords, dims=([3],[0])) # (B,N,3,nx,ny,nz)
        projected_coords = projected_norm_coords[:,:,:2,:,:,:] / projected_norm_coords[:,:,2:3,:,:,:] # (B,N,2,nx,ny,nz)
        backbone_coords = (projected_coords/4).to(torch.int32) # (B,N,2,nx,ny,nz), backbone output pixel coords
        ib, inn, ix, iy, iz = torch.meshgrid([torch.arange(x.images.shape[0]), torch.arange(x.images.shape[1]), torch.arange(self.bev_norm_coords.shape[1]), torch.arange(self.bev_norm_coords.shape[2]), torch.arange(self.bev_norm_coords.shape[3])])
        camera_mask = (projected_norm_coords[:,:,2,:,:] >= 0) & \
                    (backbone_coords[:,:,0,:,:,:] >= 0) & \
                    (backbone_coords[:,:,1,:,:,:] >= 0) & \
                    (backbone_coords[:,:,0,:,:,:] >= 0) &  \
                    (backbone_coords[:,:,0,:,:,:] < backbone_out.shape[2]) & \
                    (backbone_coords[:,:,1,:,:,:] < backbone_out.shape[3])
        bev_from_backbone = torch.where(camera_mask.unsqueeze(2), backbone_out[ib, inn, :, backbone_coords[:,:,1,:,:,:].clamp(0,backbone_out.shape[3]-1), backbone_coords[:,:,0,:,:,:].clamp(0,backbone_out.shape[2]-1)].movedim(-1,2), 0.0) # (B,N,128,nx,ny,nz)
        return (bev_from_backbone.sum(dim=1)/camera_mask.sum(dim=1).unsqueeze(1).clamp(min=1)).sum(dim=4) # mean over cameras, sum over z

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
    def __init__(self, bev_raster_spec: BEVGridSpec, bev_raster_labels: tuple[str, ...], min_z : float, max_z: float, num_z: int):
        super().__init__()
        self.bevlift = BEVLift(ResNetBackbone(), bev_raster_spec, min_z, max_z, num_z)
        self.bevseg = BEVSeg(bev_raster_labels)
    
    def forward(self, x: CameraDataBatch):
        return self.bevseg(self.bevlift(x))