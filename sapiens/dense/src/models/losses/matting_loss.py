# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import logging
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from .....registry import MODELS  # pyre-ignore[21]
from torch import Tensor

from .utils import weight_reduce_loss

logger: logging.Logger = logging.getLogger(__name__)


class BaseMattingLoss(nn.Module):
    """Base class for all Matting loss functions."""

    def __init__(self, loss_name: str, loss_weight: float) -> None:
        super().__init__()
        self._loss_name = loss_name
        self._loss_weight = loss_weight

    @property
    def loss_name(self) -> str:
        """Returns the name of this loss function."""
        return self._loss_name


@MODELS.register_module()
class MattingL1Loss(BaseMattingLoss):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        eps: float = -100,
        loss_name: str = "loss_l1",
    ) -> None:
        super().__init__(loss_name, loss_weight)
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )
        assert reduction_override in (
            None,
            "none",
            "mean",
            "sum",
        ), "Invalid reduction_override value"
        if valid_mask is not None and valid_mask.sum() == 0:
            return torch.tensor(0, dtype=pred.dtype, device=pred.device)
        reduction = reduction_override if reduction_override else self.reduction

        loss = F.l1_loss(pred, target, reduction="none")  # B x C x H x W
        if valid_mask is not None:
            loss *= valid_mask

        loss = (
            weight_reduce_loss(loss, weight, reduction, avg_factor) * self._loss_weight
        )

        # Convert nan to 0
        # torch.nan_to_num expects nan, posinf, neginf as Number
        loss = torch.nan_to_num(
            loss,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return loss


def compute_sobel_gradients(x: Tensor) -> tuple[Tensor, Tensor]:
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)

    B, C, H, W = x.shape
    x = x.reshape(B * C, 1, H, W)

    # Apply padding to maintain size
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")

    # Compute gradients using Sobel filters
    grad_x = F.conv2d(x_pad, sobel_x, padding=0)
    grad_y = F.conv2d(x_pad, sobel_y, padding=0)

    grad_x = grad_x.reshape(B, C, H, W)
    grad_y = grad_y.reshape(B, C, H, W)

    return grad_x, grad_y


@MODELS.register_module()
class MattingGradLoss(BaseMattingLoss):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_grad",
    ) -> None:
        super().__init__(loss_name, loss_weight)
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )
        assert reduction_override in (
            None,
            "none",
            "mean",
            "sum",
        ), "Invalid reduction_override value"
        if valid_mask is not None and valid_mask.sum() == 0:
            return torch.tensor(0, dtype=pred.dtype, device=pred.device)

        reduction = reduction_override if reduction_override else self.reduction

        pred_grad_x, pred_grad_y = compute_sobel_gradients(pred)
        gt_grad_x, gt_grad_y = compute_sobel_gradients(target)
        pred_grad_mag = torch.sqrt(pred_grad_x.pow(2) + pred_grad_y.pow(2) + 1e-6)
        gt_grad_mag = torch.sqrt(gt_grad_x.pow(2) + gt_grad_y.pow(2) + 1e-6)

        loss_grad = F.l1_loss(pred_grad_mag, gt_grad_mag, reduction="none")
        if valid_mask is not None:
            loss_grad *= valid_mask

        loss = (
            weight_reduce_loss(loss_grad, weight, reduction, avg_factor)
            * self._loss_weight
        )

        return loss


@lru_cache(maxsize=1000)
def gauss_kernel(
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> Tensor:
    kernel = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        device=device,
        dtype=dtype,
    )
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel


def laplacian_loss(
    pred: Tensor,
    target: Tensor,
    max_levels: int,
    reduction: str,
    valid_mask: Tensor | None = None,
) -> Tensor:
    pred_pyramid = laplacian_pyramid(pred, max_levels)
    target_pyramid = laplacian_pyramid(target, max_levels)
    loss: Tensor = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for level in range(max_levels):
        if valid_mask is not None:
            mask = F.interpolate(
                input=valid_mask.float(),
                size=pred_pyramid[level].shape[2:],
                mode="nearest",
            )

            loss += (2**level) * F.l1_loss(
                pred_pyramid[level] * mask,
                target_pyramid[level] * mask,
                reduction=reduction,
            )
        else:
            loss += (2**level) * F.l1_loss(
                pred_pyramid[level], target_pyramid[level], reduction=reduction
            )

    if reduction == "mean":
        loss = loss / max_levels
    return loss


def laplacian_pyramid(img: Tensor, max_levels: int) -> list[Tensor]:
    kernel = gauss_kernel(device=img.device, dtype=img.dtype)
    current = img
    pyramid: list[Tensor] = []
    for _ in range(max_levels - 1):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    # Append the top level of the pyramid (the final non-difference level)
    pyramid.append(current)
    return pyramid


def gauss_convolution(img: Tensor, kernel: Tensor) -> Tensor:
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode="reflect")
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img


def downsample(img: Tensor, kernel: Tensor) -> Tensor:
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img


def upsample(img: Tensor, kernel: Tensor) -> Tensor:
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out


def crop_to_even_size(img: Tensor) -> Tensor:
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]


@MODELS.register_module()
class MattingLaplacianLoss(BaseMattingLoss):
    def __init__(
        self,
        max_levels: int = 5,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_lap",
    ) -> None:
        super().__init__(loss_name, loss_weight)
        self.max_levels = max_levels
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )
        assert reduction_override in (
            None,
            "none",
            "mean",
            "sum",
        ), "Invalid reduction_override value"
        if valid_mask is not None and valid_mask.sum() == 0:
            return torch.tensor(0, dtype=pred.dtype, device=pred.device)

        reduction = reduction_override if reduction_override else self.reduction

        loss = laplacian_loss(
            pred,
            target,
            valid_mask=valid_mask,
            max_levels=self.max_levels,
            reduction=reduction,
        )

        loss = (
            weight_reduce_loss(loss, weight, reduction, avg_factor) * self._loss_weight
        )

        return loss


def compute_rolling_block(
    A: npt.NDArray[np.int_], block: tuple[int, int] = (3, 3)
) -> npt.NDArray[np.int_]:
    """Compute rolling blocks of a 2D array.

    Args:
        A: Input 2D array
        block: Block size as (height, width)

    Returns:
        Array of rolling blocks
    """
    shape = ((A.shape[0] - block[0] + 1), (A.shape[1] - block[1] + 1)) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)


def prepare_window_indices(
    index_matrix: npt.NDArray[np.int_],
    height: int,
    width: int,
    window_radius: int,
) -> tuple[npt.NDArray[np.int_], int, int]:
    """Prepare window indices for the matting Laplacian.

    Args:
        index_matrix: Matrix of flattened indices
        height: Image height
        width: Image width
        window_radius: Radius of the local window

    Returns:
        Window indices array
    """
    window_diameter = 2 * window_radius + 1
    window_size = window_diameter**2

    # Number of valid window center positions
    center_height = height - 2 * window_radius
    center_width = width - 2 * window_radius

    # Extract sliding window indices
    window_indices = compute_rolling_block(
        index_matrix, block=(window_diameter, window_diameter)
    )
    window_indices = window_indices.reshape(center_height, center_width, window_size)
    return window_indices, window_size, window_diameter


def apply_mask_to_windows(
    indices: npt.NDArray[np.int_],
    mask: npt.NDArray[np.uint8] | None,
    diameter: int,
) -> npt.NDArray[np.int_]:
    """Apply mask to window indices if provided.

    Args:
        indices: Array of window indices
        mask: Binary mask
        diameter: Diameter of the window

    Returns:
        Masked window indices
    """
    if mask is not None:
        dilated_mask = cv2.dilate(
            mask.astype(np.uint8), np.ones((diameter, diameter), np.uint8)
        ).astype(np.bool_)
        window_mask_sum = np.sum(dilated_mask.ravel()[indices], axis=2)
        return indices[window_mask_sum > 0, :]
    else:
        return indices.reshape(-1, indices.shape[-1])


def compute_window_statistics(
    window_values: npt.NDArray[np.floating[Any]],
    window_size: int,
    eps: float,
    num_channels: int,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Compute statistics for each window.

    Args:
        window_values: Image values within each window
        window_size: Size of each window
        eps: Regularization parameter
        num_channels: Number of image channels

    Returns:
        Window statistics needed for Laplacian computation
    """
    # Compute window statistics
    window_mean = np.mean(window_values, axis=1, keepdims=True)
    window_covariance = (
        window_values.swapaxes(-2, -1) @ window_values
    ) / window_size - (window_mean.swapaxes(-2, -1) @ window_mean)

    # Compute inverse of regularized covariance
    regularizer = (eps / window_size) * np.eye(num_channels)
    covariance_inv = np.linalg.inv(window_covariance + regularizer)

    return window_mean, covariance_inv


def compute_laplacian_values(
    window_values: npt.NDArray[np.floating[Any]],
    window_mean: npt.NDArray[np.floating[Any]],
    covariance_inv: npt.NDArray[np.floating[Any]],
    window_size: int,
) -> npt.NDArray[np.floating[Any]]:
    """Compute Laplacian values for each window.

    Args:
        window_values: Image values within each window
        window_mean: Mean of each window
        covariance_inv: Inverse of regularized covariance
        window_size: Size of each window

    Returns:
        Laplacian values
    """
    centered_values = window_values - window_mean
    transformed = centered_values @ covariance_inv
    return np.eye(window_size) - (1.0 / window_size) * (
        1 + transformed @ centered_values.transpose(0, 2, 1)
    )


def create_sparse_tensor(
    window_indices: npt.NDArray[np.int_],
    laplacian_values: npt.NDArray[np.floating[Any]],
    height: int,
    width: int,
) -> torch.Tensor:
    """Create sparse COO tensor from Laplacian values.

    Args:
        window_indices: Array of window indices
        laplacian_values: Computed Laplacian values
        height: Image height
        width: Image width

    Returns:
        Sparse COO tensor
    """
    window_size = window_indices.shape[1]
    # Build sparse matrix indices
    col_indices = np.tile(window_indices, window_size).ravel()
    row_indices = np.repeat(window_indices, window_size).ravel()
    values = laplacian_values.ravel()

    # Create sparse COO tensor
    indices = torch.tensor(np.stack([row_indices, col_indices]), dtype=torch.long)
    sparse_values = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices, sparse_values, size=(height * width, height * width)
    )


def compute_matting_laplacian(
    image: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.uint8] | None = None,
    eps: float = 1e-7,
    window_radius: int = 1,
) -> torch.Tensor:
    """Compute the Matting Laplacian matrix for a given image.

    This implements the closed-form matting Laplacian from:
    "A Closed Form Solution to Natural Image Matting" by Levin et al.

    Args:
        image: Input image array of shape (H, W, C) with C channels (typically 3).
        mask: Optional binary mask of shape (H, W). If provided, the Laplacian
            is computed only for pixels where mask is True. If None, computed
            for all pixels.
        eps: Regularization parameter controlling alpha smoothness.
            Corresponds to epsilon in Eq. 12 of the paper.
        window_radius: Radius of the local window used to build the Laplacian.
            Window size will be (2 * window_radius + 1)^2.

    Returns:
        Sparse COO tensor of shape (H*W, H*W) holding the Matting Laplacian.
    """
    height, width, num_channels = image.shape

    # Create index matrix and flatten image
    index_matrix = np.arange(height * width).reshape((height, width))
    flat_image = image.reshape(height * width, num_channels)

    # Prepare window indices
    window_indices, window_size, window_diameter = prepare_window_indices(
        index_matrix, height, width, window_radius
    )

    # Apply mask if provided
    window_indices = apply_mask_to_windows(window_indices, mask, window_diameter)

    # Get image values within each window: (num_windows, window_size, num_channels)
    window_values = flat_image[window_indices]

    # Compute window statistics
    window_mean, covariance_inv = compute_window_statistics(
        window_values, window_size, eps, num_channels
    )

    # Compute Laplacian values
    laplacian_values = compute_laplacian_values(
        window_values, window_mean, covariance_inv, window_size
    )

    # Create sparse tensor
    return create_sparse_tensor(window_indices, laplacian_values, height, width)


def scipy_sparse_to_torch_sparse(
    sparse_matrix: Any,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert scipy sparse COO matrix to PyTorch sparse tensor."""
    # Ensure COO format
    coo = sparse_matrix.tocoo()
    # Create indices tensor (2 x nnz)
    indices = torch.tensor(
        [coo.row, coo.col],
        dtype=torch.long,
        device=device,
    )
    # Create values tensor
    values = torch.tensor(
        coo.data,
        dtype=dtype,
        device=device,
    )
    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        device=device,
        dtype=dtype,
    )
    return sparse_tensor


def closed_form_matting_with_prior(
    image: npt.NDArray[np.floating[Any]],
    prior: npt.NDArray[np.floating[Any]],
    prior_confidence: npt.NDArray[np.floating[Any]],
    consts_map: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Applies closed form matting with prior alpha map to image.

    Args:
        image: 3-dim numpy matrix with input image of shape (H, W, C).
        prior: matrix of same width and height as input image holding apriori alpha map.
        prior_confidence: matrix of the same shape as prior holding confidence of prior alpha.
        consts_map: binary mask of pixels that aren't expected to change due to high
            prior confidence.

    Returns:
        2-dim matrix holding computed alpha map.
    """
    assert image.shape[:2] == prior.shape, (
        "prior must be 2D matrix with height and width equal to image."
    )
    assert image.shape[:2] == prior_confidence.shape, (
        "prior_confidence must be 2D matrix with height and width equal to image."
    )
    if consts_map is not None:
        assert image.shape[:2] == consts_map.shape, (
            "consts_map must be 2D matrix with height and width equal to image."
        )

    logger.info("Computing Matting Laplacian.")
    laplacian = compute_matting_laplacian(
        image, ~consts_map if consts_map is not None else None
    )

    # Convert sparse torch tensor to scipy sparse for solving
    laplacian_np = laplacian.coalesce()
    indices = laplacian_np.indices().cpu().numpy()
    values = laplacian_np.values().cpu().numpy()
    laplacian_scipy = sp_sparse.coo_matrix(
        (values, (indices[0], indices[1])), shape=laplacian_np.shape
    ).tocsr()

    confidence = sp_sparse.diags(prior_confidence.ravel())
    logger.info("Solving for alpha.")
    solution = sp_linalg.spsolve(
        laplacian_scipy + confidence, prior.ravel() * prior_confidence.ravel()
    )
    alpha = np.minimum(np.maximum(solution.reshape(prior.shape), 0), 1)
    return alpha


def closed_form_matting_with_trimap(
    image: npt.NDArray[np.floating[Any]],
    trimap: npt.NDArray[np.floating[Any]],
    trimap_confidence: float = 100.0,
) -> npt.NDArray[np.floating[Any]]:
    """Apply Closed-Form matting to given image using trimap.

    Args:
        image: Input image of shape (H, W, C).
        trimap: Trimap of shape (H, W) with values in [0, 1].
        trimap_confidence: Confidence value for trimap constraints.

    Returns:
        Computed alpha matte of shape (H, W).
    """
    assert image.shape[:2] == trimap.shape, (
        "trimap must be 2D matrix with height and width equal to image."
    )
    consts_map = (trimap < 0.1) | (trimap > 0.9)
    return closed_form_matting_with_prior(
        image, trimap, trimap_confidence * consts_map, consts_map
    )


@MODELS.register_module()
class MattingNaturalImageLaplacianLoss(BaseMattingLoss):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_natural_image_lap",
    ) -> None:
        super().__init__(loss_name, loss_weight)
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        image: Tensor,
        valid_mask: Tensor | None = None,
        weight: Tensor | None = None,
    ) -> Tensor:
        """Compute the natural image matting Laplacian loss.

        Args:
            pred: Predicted alpha matte of shape (H, W) or (H, W, 1).
                Must be single-channel as the Laplacian loss operates on alpha values.
            image: Input RGB image of shape (H, W, C) used to compute the Laplacian.
                The Laplacian encodes color-based smoothness constraints.
            valid_mask: Optional mask of shape (H, W) or (H, W, 1).
            weight: Unused, kept for API consistency.

        Returns:
            Scalar loss value: alpha^T @ L @ alpha
        """
        # Ensure pred is 2D (H, W) for the Laplacian computation
        if pred.dim() == 3:
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            else:
                raise ValueError(
                    f"pred must be single-channel (H, W) or (H, W, 1), got {pred.shape}"
                )

        height, width = pred.shape
        assert image.shape[0] == height and image.shape[1] == width, (
            f"Spatial dimensions of pred ({pred.shape}) and image ({image.shape}) must match"
        )

        if valid_mask is not None and valid_mask.sum() == 0:
            return torch.tensor(0, dtype=pred.dtype, device=pred.device)

        # Convert to numpy and compute Laplacian
        image_np = image.detach().cpu().numpy()
        laplacian = compute_matting_laplacian(image_np)
        laplacian = laplacian.to(device=pred.device, dtype=pred.dtype)

        # Compute loss: alpha^T @ L @ alpha
        # pred is (H, W), flatten to (H*W,) to match Laplacian size (H*W, H*W)
        alpha_flat = pred.flatten()
        loss = (
            alpha_flat @ torch.sparse.mm(laplacian, alpha_flat.unsqueeze(1)).squeeze()
        ) * self._loss_weight
        return loss
