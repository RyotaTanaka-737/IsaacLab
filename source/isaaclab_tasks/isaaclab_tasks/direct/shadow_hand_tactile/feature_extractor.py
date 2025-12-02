# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self):
        super().__init__()
        num_channel = 7
        in_channel = 325
        mlp_out_dim = 128
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 58, 58]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 28, 28]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 13, 13]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 6, 6]),
            nn.AvgPool2d(6),
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        # x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        # x[:, 4:7, :, :] = self.data_transforms(x[:, 4:7, :, :])
        # cnn_x = self.cnn(x)
        # out = self.linear(cnn_x.view(-1, 128))
        # print(x)
        out = self.mlp(x)
        return out


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str, log_dir: str | None = None):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            log_dir: Directory to save checkpoints. If None, uses local "logs" folder resolved with respect to this file.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork()
        self.feature_extractor.to("cuda:0")

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            # print('train')
            self.feature_extractor.train()
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            # self.feature_extractor.train()
            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = True
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed RGB, depth, and segmentation
        """
        rgb_img = rgb_img / 255.0
        # process depth image
        depth_img[depth_img == float("inf")] = 0
        depth_img /= 5.0
        depth_img /= torch.max(depth_img)
        # process segmentation image
        segmentation_img = segmentation_img / 255.0
        mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
        segmentation_img -= mean_tensor
        return rgb_img, depth_img, segmentation_img

    def _save_images(self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3).
        """
        save_images_to_file(rgb_img, "shadow_hand_rgb.png")
        save_images_to_file(depth_img, "shadow_hand_depth.png")
        save_images_to_file(segmentation_img, "shadow_hand_segmentation.png")

    def step(
        self, gt_feature: torch.Tensor, input_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            gt_feature (torch.Tensor): Ground truth feature tensor. Shape: (N, 27).
            input_obs (torch.Tensor): Input observation tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature loss and predicted feature.
        """
        self.feature_extractor.train()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            # print(param)
        self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
        current_obs = input_obs.clone().float().to("cuda:0")
        if current_obs.numel() == 0:
            print('NUUUUUUUUUUUUUUUUUUUL')
        if self.cfg.train:
            with torch.enable_grad():
                # with torch.inference_mode(False):
                self.optimizer.zero_grad()

                predicted_feature = self.feature_extractor(current_obs)
                # print('grad_fn', predicted_feature.grad_fn)
                feature_loss = self.l2_loss(predicted_feature, gt_feature.clone().float()) * 100
                if feature_loss.requires_grad and feature_loss.grad_fn is not None:
                    feature_loss.backward()
                    self.optimizer.step()
                else:
                    print("Warning: Skipping backward pass (No grad_fn found this step).")

                if self.step_count % 50000 == 0:
                    torch.save(
                        self.feature_extractor.state_dict(),
                        os.path.join(self.log_dir, f"cnn_{self.step_count}_{feature_loss.detach().cpu().numpy()}.pth"),
                    )

                self.step_count += 1

                return feature_loss, predicted_feature
        else:
            predicted_feature = self.feature_extractor(input_obs)
            return None, predicted_feature
