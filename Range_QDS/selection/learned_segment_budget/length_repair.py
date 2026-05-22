"""Length-aware point selection and repair helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from selection.retained_mask_selectors import deterministic_topk_with_jitter
from workloads.range_geometry import local_equirectangular_distance_km


@dataclass
class _LengthRepairState:
    trajectory_id: int
    start: int
    end: int
    points: torch.Tensor
    scores: torch.Tensor
    retained: torch.Tensor
    learned: torch.Tensor
    fallback: torch.Tensor
    repair: torch.Tensor
    removable: torch.Tensor
    swaps: int = 0


@dataclass(frozen=True)
class _LengthRepairCandidate:
    state_index: int
    add_idx: int
    remove_idx: int
    net_gain: float
    candidate_key: float
    removal_key: float


def _normalize_candidate_values(values: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    """Min-max normalize finite candidate values, keeping invalid entries at -inf."""
    out = torch.full_like(values.float(), -float("inf"))
    if not bool(finite.any().item()):
        return out
    finite_values = values.float()[finite]
    min_value = finite_values.min()
    span = finite_values.max() - min_value
    if float(span.item()) <= 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (finite_values - min_value) / span
    return out


def _local_distance_km(
    local_points: torch.Tensor, left_idx: torch.Tensor, right_idx: torch.Tensor
) -> torch.Tensor:
    """Return approximate lat/lon distance in km for local index pairs."""
    left = local_points[left_idx.long()]
    right = local_points[right_idx.long()]
    lat1 = left[:, 1].float()
    lon1 = left[:, 2].float()
    lat2 = right[:, 1].float()
    lon2 = right[:, 2].float()
    return local_equirectangular_distance_km(lat1, lon1, lat2, lon2)


def _length_gain_scores(
    local_points: torch.Tensor | None,
    retained_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
) -> torch.Tensor:
    """Return path-length gain from adding each candidate between retained neighbors."""
    if local_points is None or int(local_points.shape[0]) != int(candidate_scores.numel()):
        return torch.zeros_like(candidate_scores.float())
    finite = torch.isfinite(candidate_scores)
    retained_sorted = retained_indices.to(device=candidate_scores.device, dtype=torch.long).unique(
        sorted=True
    )
    if int(retained_sorted.numel()) < 2 or not bool(finite.any().item()):
        return torch.zeros_like(candidate_scores.float())
    candidate_idx = torch.arange(int(candidate_scores.numel()), device=candidate_scores.device)
    pos = torch.searchsorted(retained_sorted, candidate_idx)
    valid = finite & (pos > 0) & (pos < int(retained_sorted.numel()))
    gains = torch.zeros_like(candidate_scores.float())
    if not bool(valid.any().item()):
        return gains
    valid_idx = candidate_idx[valid]
    valid_pos = pos[valid]
    left_idx = retained_sorted[valid_pos - 1]
    right_idx = retained_sorted[valid_pos]
    local_points_device = local_points.to(device=candidate_scores.device)
    via_candidate = _local_distance_km(
        local_points_device, left_idx, valid_idx
    ) + _local_distance_km(local_points_device, valid_idx, right_idx)
    shortcut = _local_distance_km(local_points_device, left_idx, right_idx)
    gains[valid] = torch.clamp(via_candidate - shortcut, min=0.0)
    return gains


def _length_loss_scores(
    local_points: torch.Tensor | None,
    retained_indices: torch.Tensor,
    removable_indices: torch.Tensor,
) -> torch.Tensor:
    """Return path-length loss from removing retained candidate indices."""
    if (
        local_points is None
        or int(retained_indices.numel()) < 3
        or int(removable_indices.numel()) <= 0
        or int(local_points.shape[0]) <= 0
    ):
        return torch.full((int(removable_indices.numel()),), float("inf"), dtype=torch.float32)
    retained_sorted = retained_indices.to(dtype=torch.long).unique(sorted=True)
    removable = removable_indices.to(device=retained_sorted.device, dtype=torch.long)
    pos = torch.searchsorted(retained_sorted, removable)
    valid = (pos > 0) & (pos < int(retained_sorted.numel()) - 1)
    losses = torch.full(
        (int(removable.numel()),), float("inf"), dtype=torch.float32, device=retained_sorted.device
    )
    if not bool(valid.any().item()):
        return losses.cpu()
    valid_removable = removable[valid]
    valid_pos = pos[valid]
    left_idx = retained_sorted[valid_pos - 1]
    right_idx = retained_sorted[valid_pos + 1]
    local_points_device = local_points.to(device=retained_sorted.device)
    via_removed = _local_distance_km(
        local_points_device, left_idx, valid_removable
    ) + _local_distance_km(local_points_device, valid_removable, right_idx)
    shortcut = _local_distance_km(local_points_device, left_idx, right_idx)
    losses[valid] = torch.clamp(via_removed - shortcut, min=0.0)
    return losses.cpu()


def _fill_missing_by_length_gain(
    *,
    retained: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    budget: int,
) -> int:
    """Fill any diagnostic under-budget slots by query-free path-length gain."""
    missing = max(0, int(budget) - int(retained.sum().item()))
    if missing <= 0:
        return 0
    filled = 0
    points_cpu = points.detach().cpu().float()
    for trajectory_id, (start, end) in enumerate(boundaries):
        if filled >= missing:
            break
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 3:
            continue
        local_retained = retained[start_i:end_i].detach().cpu().bool()
        retained_indices = torch.where(local_retained)[0]
        if int(retained_indices.numel()) < 2:
            continue
        candidate_scores = torch.zeros((end_i - start_i,), dtype=torch.float32)
        candidate_scores[local_retained] = -float("inf")
        while filled < missing:
            finite = torch.isfinite(candidate_scores)
            if not bool(finite.any().item()):
                break
            retained_indices = torch.where(local_retained)[0]
            gain_scores = _length_gain_scores(
                points_cpu[start_i:end_i], retained_indices, candidate_scores
            )
            positive_gain = finite & (gain_scores > 1e-9)
            if not bool(positive_gain.any().item()):
                break
            gain_scores[~positive_gain] = -float("inf")
            choice = deterministic_topk_with_jitter(
                gain_scores,
                1,
                trajectory_id * 11003 + filled,
            )
            if int(choice.numel()) == 0:
                break
            idx = int(choice[0].item())
            local_retained[idx] = True
            candidate_scores[idx] = -float("inf")
            retained[start_i + idx] = True
            filled += 1
    return int(filled)


def _apply_length_repair_swaps(
    *,
    scores: torch.Tensor,
    points: torch.Tensor | None,
    boundaries: list[tuple[int, int]],
    retained: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
    repair_fraction: float,
    score_protection_fraction: float = 0.0,
    length_repair_protected_mask: torch.Tensor | None = None,
) -> int:
    """Swap a bounded global share of learned slots toward query-free path-length gain."""
    if (
        points is None
        or int(points.shape[0]) != int(scores.numel())
        or float(repair_fraction) <= 0.0
    ):
        return 0
    fraction = max(0.0, min(1.0, float(repair_fraction)))
    points_cpu = points.detach().cpu().float()
    scores_cpu = scores.detach().cpu().float()
    retained_cpu = retained.detach().cpu().bool()
    global_removable = learned_mask.detach().cpu().bool() | fallback_mask.detach().cpu().bool()
    global_removable &= retained_cpu
    protected_cpu = _score_protected_repair_mask(
        scores=scores_cpu,
        retained=retained_cpu,
        removable=global_removable,
        protection_fraction=float(score_protection_fraction),
    )
    global_removable &= ~protected_cpu
    if length_repair_protected_mask is not None:
        length_repair_protected_mask[:] = protected_cpu.to(
            device=length_repair_protected_mask.device
        )
    states: list[_LengthRepairState] = []
    total_removable = 0

    for trajectory_id, (start, end) in enumerate(boundaries):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 3:
            continue
        local_retained = retained_cpu[start_i:end_i].clone()
        local_learned = learned_mask[start_i:end_i].detach().cpu().bool().clone()
        local_fallback = fallback_mask[start_i:end_i].detach().cpu().bool().clone()
        local_repair = length_repair_mask[start_i:end_i].detach().cpu().bool().clone()
        local_removable = global_removable[start_i:end_i].clone()
        removable_count = int(local_removable.sum().item())
        if removable_count <= 0:
            continue
        total_removable += removable_count
        states.append(
            _LengthRepairState(
                trajectory_id=int(trajectory_id),
                start=start_i,
                end=end_i,
                points=points_cpu[start_i:end_i],
                scores=scores_cpu[start_i:end_i],
                retained=local_retained,
                learned=local_learned,
                fallback=local_fallback,
                repair=local_repair,
                removable=local_removable,
            )
        )

    max_total_swaps = min(total_removable, math.ceil(fraction * float(total_removable)))
    total_swaps = 0
    best_by_state = {
        idx: candidate
        for idx, state in enumerate(states)
        if (candidate := _best_length_repair_candidate(idx, state)) is not None
    }
    while total_swaps < max_total_swaps and best_by_state:
        _idx, candidate = max(
            best_by_state.items(),
            key=lambda item: _length_repair_candidate_sort_key(item[1], states[item[0]]),
        )
        state = states[candidate.state_index]
        _apply_length_repair_candidate(state, candidate)
        total_swaps += 1
        next_candidate = _best_length_repair_candidate(candidate.state_index, state)
        if next_candidate is None:
            best_by_state.pop(candidate.state_index, None)
        else:
            best_by_state[candidate.state_index] = next_candidate

    for state in states:
        retained[state.start : state.end] = state.retained.to(device=retained.device)
        learned_mask[state.start : state.end] = state.learned.to(device=learned_mask.device)
        fallback_mask[state.start : state.end] = state.fallback.to(device=fallback_mask.device)
        length_repair_mask[state.start : state.end] = state.repair.to(
            device=length_repair_mask.device
        )

    return int(total_swaps)


def _score_protected_repair_mask(
    *,
    scores: torch.Tensor,
    retained: torch.Tensor,
    removable: torch.Tensor,
    protection_fraction: float,
) -> torch.Tensor:
    """Return removable retained points protected from length-repair removal."""
    protected = torch.zeros_like(removable, dtype=torch.bool)
    fraction = max(0.0, min(1.0, float(protection_fraction)))
    if fraction <= 0.0 or int(removable.numel()) <= 0:
        return protected
    retained_count = int(retained.sum().item())
    finite_protectable = removable.bool() & torch.isfinite(scores.float())
    protectable_count = int(finite_protectable.sum().item())
    if retained_count <= 0 or protectable_count <= 0:
        return protected
    protect_count = min(protectable_count, math.ceil(fraction * float(retained_count)))
    if protect_count <= 0:
        return protected
    candidate_scores = scores.float().clone()
    candidate_scores[~finite_protectable] = -float("inf")
    protected_idx = deterministic_topk_with_jitter(candidate_scores, protect_count, 760711)
    if int(protected_idx.numel()) > 0:
        protected[protected_idx.long()] = True
    return protected


def _best_length_repair_candidate(
    state_index: int, state: _LengthRepairState
) -> _LengthRepairCandidate | None:
    """Return the best positive net path-length swap for one trajectory."""
    retained_indices = torch.where(state.retained)[0]
    if int(retained_indices.numel()) < 3:
        return None
    candidate_scores = state.scores.clone()
    candidate_scores[state.retained] = -float("inf")
    finite_candidates = torch.isfinite(candidate_scores)
    if not bool(finite_candidates.any().item()):
        return None
    gain_scores = _length_gain_scores(state.points, retained_indices, candidate_scores)
    positive_gain = finite_candidates & (gain_scores > 1e-9)
    if not bool(positive_gain.any().item()):
        return None
    normalized_gain = _normalize_candidate_values(gain_scores, positive_gain)
    normalized_score = _normalize_candidate_values(candidate_scores, positive_gain)
    candidate_key = 0.90 * normalized_gain + 0.10 * normalized_score
    candidate_key[~positive_gain] = -float("inf")
    add_idx_tensor = deterministic_topk_with_jitter(
        candidate_key,
        1,
        state.trajectory_id * 65537 + state.swaps,
    )
    if int(add_idx_tensor.numel()) <= 0:
        return None
    add_idx = int(add_idx_tensor[0].item())

    removable_indices = torch.where(state.removable & state.retained)[0]
    if int(removable_indices.numel()) <= 0:
        return None
    removal_losses = _length_loss_scores(state.points, retained_indices, removable_indices)
    finite_removable = torch.isfinite(removal_losses)
    if not bool(finite_removable.any().item()):
        return None
    removable_scores = state.scores[removable_indices].float()
    normalized_loss = _normalize_candidate_values(removal_losses, finite_removable)
    normalized_removable_score = _normalize_candidate_values(removable_scores, finite_removable)
    removal_key = (1.0 - normalized_loss) + 0.10 * (1.0 - normalized_removable_score)
    removal_key[~finite_removable] = -float("inf")
    remove_choice = deterministic_topk_with_jitter(
        removal_key,
        1,
        state.trajectory_id * 91733 + state.swaps,
    )
    if int(remove_choice.numel()) <= 0:
        return None
    remove_choice_position = int(remove_choice[0].item())
    remove_idx = int(removable_indices[remove_choice_position].item())
    add_gain = float(gain_scores[add_idx].item())
    removal_loss = float(removal_losses[remove_choice_position].item())
    net_gain = add_gain - removal_loss
    if net_gain <= 1e-9:
        return None
    return _LengthRepairCandidate(
        state_index=int(state_index),
        add_idx=add_idx,
        remove_idx=remove_idx,
        net_gain=net_gain,
        candidate_key=float(candidate_key[add_idx].item()),
        removal_key=float(removal_key[remove_choice_position].item()),
    )


def _length_repair_candidate_sort_key(
    candidate: _LengthRepairCandidate, state: _LengthRepairState
) -> tuple[float, float, float, int, int]:
    """Sort candidates by global length gain with deterministic tie-breaks."""
    return (
        float(candidate.net_gain),
        float(candidate.candidate_key),
        float(candidate.removal_key),
        -int(state.trajectory_id),
        -int(candidate.add_idx),
    )


def _apply_length_repair_candidate(
    state: _LengthRepairState, candidate: _LengthRepairCandidate
) -> None:
    """Apply one precomputed local repair candidate."""
    state.retained[candidate.remove_idx] = False
    state.learned[candidate.remove_idx] = False
    state.fallback[candidate.remove_idx] = False
    state.removable[candidate.remove_idx] = False
    state.retained[candidate.add_idx] = True
    state.repair[candidate.add_idx] = True
    state.swaps += 1


def _select_with_spacing(
    local_scores: torch.Tensor,
    keep_count: int,
    *,
    trajectory_id: int,
    existing_indices: torch.Tensor,
    min_spacing: int,
    local_points: torch.Tensor | None = None,
    geometry_gain_weight: float = 0.05,
    segment_aux_scores: torch.Tensor | None = None,
    segment_score_weight: float = 0.0,
) -> torch.Tensor:
    """Select top scores with simple non-maximum spacing."""
    keep = max(0, min(int(keep_count), int(local_scores.numel())))
    if keep <= 0:
        return torch.empty((0,), dtype=torch.long, device=local_scores.device)
    candidate_scores = local_scores.clone()
    if int(existing_indices.numel()) > 0:
        candidate_scores[
            existing_indices.to(device=local_scores.device, dtype=torch.long)
        ] = -float("inf")
    selected: list[torch.Tensor] = []
    retained_indices = existing_indices.to(device=local_scores.device, dtype=torch.long)
    spacing = max(0, int(min_spacing))
    for step in range(keep):
        finite = torch.isfinite(candidate_scores)
        if not bool(finite.any().item()):
            break
        segment_weight = max(0.0, min(1.0, float(segment_score_weight)))
        score_for_selection = candidate_scores.clone()
        if segment_aux_scores is not None and segment_weight > 0.0:
            segment_scores = segment_aux_scores.to(
                device=candidate_scores.device, dtype=torch.float32
            ).clone()
            segment_scores[~finite] = -float("inf")
            segment_finite = torch.isfinite(segment_scores)
            if bool(segment_finite.any().item()):
                point_scores_norm = _normalize_candidate_values(score_for_selection, finite)
                segment_scores_norm = _normalize_candidate_values(segment_scores, segment_finite)
                blended = (
                    1.0 - segment_weight
                ) * point_scores_norm + segment_weight * segment_scores_norm
                blended[~finite] = -float("inf")
                score_for_selection = blended

        gain_scores = _length_gain_scores(local_points, retained_indices, score_for_selection)
        normalized_scores = _normalize_candidate_values(score_for_selection, finite)
        normalized_gain = _normalize_candidate_values(gain_scores, finite)
        weight = max(0.0, min(1.0, float(geometry_gain_weight)))
        combined_scores = (1.0 - weight) * normalized_scores + weight * normalized_gain
        combined_scores[~finite] = -float("inf")
        choice = deterministic_topk_with_jitter(combined_scores, 1, trajectory_id * 4099 + step)
        if int(choice.numel()) == 0:
            break
        idx = int(choice[0].item())
        selected.append(choice)
        retained_indices = torch.cat([retained_indices, choice.to(dtype=torch.long)]).unique(
            sorted=True
        )
        left = max(0, idx - spacing)
        right = min(int(candidate_scores.numel()), idx + spacing + 1)
        candidate_scores[left:right] = -float("inf")
    if len(selected) < keep:
        finite = torch.isfinite(candidate_scores)
        if bool(finite.any().item()):
            gain_scores = _length_gain_scores(local_points, retained_indices, candidate_scores)
            normalized_scores = _normalize_candidate_values(candidate_scores, finite)
            normalized_gain = _normalize_candidate_values(gain_scores, finite)
            weight = max(0.0, min(1.0, float(geometry_gain_weight)))
            combined_scores = (1.0 - weight) * normalized_scores + weight * normalized_gain
            combined_scores[~finite] = -float("inf")
            fallback = deterministic_topk_with_jitter(
                combined_scores,
                keep - len(selected),
                trajectory_id * 9173 + keep,
            )
            selected.append(fallback)
        if not selected:
            return torch.empty((0,), dtype=torch.long, device=local_scores.device)
    if not selected:
        return torch.empty((0,), dtype=torch.long, device=local_scores.device)
    return torch.cat(selected).unique(sorted=True)[:keep]


fill_missing_by_length_gain = _fill_missing_by_length_gain
select_with_spacing = _select_with_spacing
