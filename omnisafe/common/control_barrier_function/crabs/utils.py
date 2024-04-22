# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for CRABS."""
import os

import requests
import torch
import torch.nn as nn
from torch import load


class Normalizer(nn.Module):
    """Normalizes input data to have zero mean and unit variance.

    Args:
        dim (int): Dimension of the input data.
        clip (float): Clip the standard deviation to this value.
    """

    def __init__(self, dim, *, clip=10) -> None:
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('n', torch.tensor(0, dtype=torch.int64))
        self.placeholder = nn.Parameter(torch.tensor(0.0), False)  # for device info (@maybe_numpy)
        self.clip = clip

    def forward(self, x, inverse=False):
        """Normalize input data.

        Args:
            x (torch.Tensor): Input data.
            inverse (bool): If True, unnormalize the data.

        Returns:
            torch.Tensor: Normalized data.
        """
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / self.std.clamp(min=1e-6)

    def update(self, data):
        """Update the normalizer with new data.

        Args:
            data (torch.Tensor): New data.
        """
        data = data - self.mean

        m = data.shape[0]
        delta = data.mean(dim=0)
        new_n = self.n + m
        new_mean = self.mean + delta * m / new_n
        new_std = torch.sqrt(
            (self.std**2 * self.n + data.var(dim=0) * m + delta**2 * self.n * m / new_n) / new_n,
        )

        self.mean.set_(new_mean.data)
        self.std.set_(new_std.data)
        self.n.set_(new_n.data)

    def fit(self, data):
        """Fit the normalizer with new data.

        Args:
            data (torch.Tensor): New data.
        """
        n = data.shape[0]
        self.n.set_(torch.tensor(n, device=self.n.device))
        self.mean.set_(data.mean(dim=0))
        self.std.set_(data.std(dim=0))


def download_model(url, destination):
    """Download model from the cloud.

    Args:
        url (str): URL of the model.
        destination (str): Path to save the model.
    """
    response = requests.get(url)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        f.write(response.content)


def get_pretrained_model(model_path, model_url, device):
    """Get pretrained model.

    If the model is not found locally, download it from the cloud.

    Args:
        model_path (str): Path to save the model.
        model_url (str): URL of the model.
        device (torch.device): Device to load the model.

    Returns:
        torch.nn.Module: Pretrained model.
    """
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        print('Model not found locally. Downloading from cloud...')
        download_model(model_url, model_path)
    else:
        print('Model found locally.')

    return load(model_path, map_location=device)
