# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from ....engine.evaluators import BaseEvaluator
from ....registry import MODELS


@MODELS.register_module()
class MattingEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def process(self, predictions: torch.Tensor, data_samples: dict, accelerator=None):
        """
        Args:
            predictions: Tensor, predicted albedo (B, 3, H_low, W_low)
            data_samples: dict with keys:
                - "gt_alpha": (B, 1, H, W)
                - "gt_foreground": (B, 3, H, W) optional
        """
        assert accelerator is not None, "evaluation process expects an accelerator"

        gt_alphas = data_samples["gt_alpha"]  # (B,1,H,W)
        gt_foregrounds = data_samples.get("gt_foreground", None)  # (B,3,H,W)
        has_foregrounds = data_samples["has_foreground"]

        # Align spatial size
        if predictions.shape[2:] != gt_alphas.shape[2:]:
            predictions = F.interpolate(
                input=predictions,
                size=gt_alphas.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        pred_alphas = predictions[:, -1:, :, :]  # [B, 1, H, W]
        pred_foregrounds = (
            predictions[:, :3, :, :] if predictions.shape[1] == 4 else None
        )  # [B, 3, H, W]

        B = predictions.shape[0]
        per_sample_vecs = []  # each: [l1_pha, N_pha, l1_fgr, N_fgr]

        for i in range(B):
            gt_alpha = gt_alphas[i][0]  # (H,W)
            pred_alpha = pred_alphas[i][0]  # (H,W)

            # --- Pixel MAE / RMSE accumulators (average over channels per pixel) ---
            l1_pha = (gt_alpha - pred_alpha).abs()
            sum_l1_pha = l1_pha.to(torch.float64).sum().unsqueeze(0)  # (1,)
            N_pha = torch.tensor(
                [float(gt_alpha.numel())], dtype=torch.float64, device=pred_alpha.device
            )

            if has_foregrounds[i]:
                gt_foreground = gt_foregrounds[i]  # (3,H,W)
                pred_foreground = pred_foregrounds[i]  # (3,H,W)

                diff_fgr = gt_foreground - pred_foreground  # (3,H,W)
                l1_fgr = diff_fgr.abs().mean(dim=0)  # (H,W)
                sum_l1_fgr = l1_fgr.to(torch.float64).sum().unsqueeze(0)  # (1,)
                N_fgr = N_pha.clone()
            else:
                sum_l1_fgr = torch.tensor(
                    [0], dtype=torch.float64, device=pred_alpha.device
                )
                N_fgr = torch.tensor([0], dtype=torch.float64, device=pred_alpha.device)

            vec = torch.cat([sum_l1_pha, N_pha, sum_l1_fgr, N_fgr], dim=0)
            per_sample_vecs.append(vec)

        pack = torch.stack(per_sample_vecs, dim=0)  # (B_local, 4)
        gpack = accelerator.gather_for_metrics(pack)  # (B_global_step, 4)
        step_totals = gpack.sum(dim=0)  # (4,)

        if accelerator.is_main_process:
            self.results.append(step_totals)
        return

    def evaluate(self, logger=None, accelerator=None) -> Dict[str, float]:
        """
        Returns:
            Dict[str, float]: {
                'l1_alpha', 'l1_foreground'
            }
        """
        assert accelerator is not None, "evaluation aggregation expects an accelerator"

        if not accelerator.is_main_process:
            self.reset()
            return {}

        if not self.results:
            if logger is not None:
                logger.info("No results to evaluate.")
            return {}

        totals = torch.stack(self.results, dim=0).sum(dim=0)  # (4,)
        sum_l1_pha, N_pha, sum_l1_fgr, N_fgr = totals

        metrics: Dict[str, float] = {}
        if N_pha.item() > 0:
            metrics["l1_alpha"] = float((sum_l1_pha / N_pha).item())
        if N_fgr.item() > 0:
            metrics["l1_foreground"] = float((sum_l1_fgr / N_fgr).item())

        table = PrettyTable()
        table.field_names = list(metrics.keys())
        table.add_row([f"{float(v):.5f}" for v in metrics.values()])
        if logger is not None:
            logger.info("\n" + table.get_string())

        self.reset()
        return metrics
