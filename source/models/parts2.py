from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# class SimpleAttentionNetwork(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         base_model = resnet34(pretrained=True)
#         self.features = nn.Sequential(*[layer for layer in base_model.children()][:-2])
#         self.attn_conv = nn.Sequential(nn.Conv2d(512, 1, 1), nn.Sigmoid())
#         self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
#         self.mask_ = None

#     def forward(self, x):
#         x = self.features(x)

#         attn = self.attn_conv(x)  # [B, 1, H, W]
#         B, _, H, W = attn.shape
#         self.mask_ = attn.detach().cpu()

#         x = x * attn
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.reshape(B, -1)

#         return self.fc(x)

# def save_attention_mask(self, x, path):
#     B = x.shape[0]
#     self.forward(x)
#     x = x.cpu() * torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
#     x = x + torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
#     fig, axs = plt.subplots(4, 2, figsize=(6, 8))
#     plt.axis("off")
#     for i in range(4):
#         axs[i, 0].imshow(x[i].permute(1, 2, 0))
#         axs[i, 1].imshow(self.mask_[i][0])
#     plt.savefig(path)
#     plt.close()
