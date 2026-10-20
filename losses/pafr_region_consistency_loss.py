from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from losses.pafr_loss import PAFRWeakLoss


def nam_region_consistency_loss(
    logits: Tensor,
    region_partition: Tensor,
) -> Tensor:
    """Penalize probability variance inside each NAMLab region.

    Region means are computed with a deterministic sorted segment reduction
    rather than CUDA scatter/atomic accumulation. The objective is unchanged:
    every pixel is pulled toward the mean probability of its own NAMLab region.
    """

    probability = torch.sigmoid(logits.float())[:, 0]

    if region_partition.ndim == 4:
        region_partition = region_partition[:, 0]

    region_partition = region_partition.long()
    batch_size = probability.shape[0]

    # Make region IDs unique across samples without assuming a fixed maximum
    # number of regions. Only the ordering matters for the segment reduction.
    region_stride = int(region_partition.max().item()) + 1
    offsets = (
        torch.arange(
            batch_size,
            device=region_partition.device,
            dtype=region_partition.dtype,
        )
        * region_stride
    ).view(batch_size, 1, 1)

    region_ids = (region_partition + offsets).reshape(-1)
    values = probability.reshape(-1)

    order = torch.argsort(region_ids, stable=True)
    sorted_ids = region_ids[order]
    sorted_values = values[order]

    _, counts = torch.unique_consecutive(
        sorted_ids,
        return_counts=True,
    )

    cumulative = torch.cumsum(sorted_values, dim=0)
    segment_ends = torch.cumsum(counts, dim=0) - 1
    cumulative_at_ends = cumulative[segment_ends]

    if cumulative_at_ends.numel() > 1:
        previous_ends = segment_ends[:-1]
        segment_sums = torch.cat(
            (
                cumulative_at_ends[:1],
                cumulative_at_ends[1:] - cumulative[previous_ends],
            )
        )
    else:
        segment_sums = cumulative_at_ends

    segment_means = segment_sums / counts.to(sorted_values.dtype)
    pixel_means = torch.repeat_interleave(segment_means, counts)

    return (sorted_values - pixel_means).square().mean()


class PAFRRegionConsistencyLoss(PAFRWeakLoss):
    def __init__(
        self,
        region_consistency_weight: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.region_consistency_weight = float(region_consistency_weight)

    def forward(
        self,
        outputs,
        scaled_outputs,
        image,
        target,
        valid_mask,
        region_partition,
    ):
        loss_dict = super().forward(
            outputs, scaled_outputs, image, target, valid_mask
        )

        region_loss = nam_region_consistency_loss(
            outputs["pred"],
            region_partition,
        )
        weighted_region_loss = self.region_consistency_weight * region_loss

        loss_dict["loss"] = loss_dict["loss"] + weighted_region_loss
        loss_dict["loss_main"] = loss_dict["loss_main"] + weighted_region_loss
        loss_dict["loss_region_consistency"] = region_loss
        return loss_dict