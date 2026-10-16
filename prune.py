"""
Pruning Logic and Mask Management for Lottery Ticket Hypothesis Experiments.

Implements:
1. Binary Mask Management across Conv2d and Linear layers.
2. Exact Global and Layer-wise Iterative Magnitude Pruning (IMP) via top-k indices.
3. Resetting surviving weights to θ0 (Winning Ticket).
4. Mandatory Control: Random re-initialization with identical mask (θ_rand ⊙ m).
5. Extra Control: Random mask generation matching exact layer-wise sparsity.
6. Strict mask integrity verification assertion.
"""

import copy
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from config import set_seed

def is_prunable_module(module: nn.Module, is_classifier: bool = False, prune_classifier: bool = False) -> bool:
    """Checks if a module's weight should be included in pruning."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if is_classifier and not prune_classifier:
            return False
        return module.weight is not None
    return False


def get_classifier_module(model: nn.Module) -> Optional[nn.Module]:
    """Retrieves the final classification layer module."""
    if hasattr(model, 'fc2'):
        return model.fc2
    elif hasattr(model, 'fc'):
        return model.fc
    return None


def init_masks(model: nn.Module, prune_classifier: bool = False) -> Dict[str, torch.Tensor]:
    """
    Initializes binary masks (all ones, shape matching layer weights)
    for all prunable modules.
    """
    classifier = get_classifier_module(model)
    masks: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        is_cls = (module is classifier)
        if is_prunable_module(module, is_classifier=is_cls, prune_classifier=prune_classifier):
            # Mask is float32 (1.0 for active, 0.0 for pruned)
            masks[name] = torch.ones_like(module.weight.data, dtype=torch.float32)
    return masks


def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    In-place projects weights onto the mask subspace: w <- w * m.
    Must be called after every optimizer.step().
    """
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name].to(module.weight.device)
            module.weight.data.mul_(mask)


def compute_actual_sparsity(masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Counts total prunable weights, surviving weights (non-zero),
    and returns exact global and per-layer sparsity statistics.
    """
    total_weights = 0
    surviving_weights = 0
    layer_sparsity = {}

    for name, mask in masks.items():
        layer_total = mask.numel()
        layer_surviving = int(mask.sum().item())
        layer_pruned = layer_total - layer_surviving
        total_weights += layer_total
        surviving_weights += layer_surviving
        layer_sparsity[name] = {
            "total": layer_total,
            "surviving": layer_surviving,
            "fraction_surviving": layer_surviving / layer_total if layer_total > 0 else 1.0,
            "fraction_pruned": layer_pruned / layer_total if layer_total > 0 else 0.0
        }

    global_fraction_surviving = surviving_weights / total_weights if total_weights > 0 else 1.0
    return {
        "total_weights": total_weights,
        "surviving_weights": surviving_weights,
        "fraction_surviving": global_fraction_surviving,
        "fraction_pruned": 1.0 - global_fraction_surviving,
        "layers": layer_sparsity
    }


def verify_mask_integrity(model: nn.Module, masks: Dict[str, torch.Tensor], tolerance: float = 1e-7) -> None:
    """
    Strict assertion ensuring no pruned weight has leaked or resurrected.
    """
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name].to(module.weight.device)
            pruned_weights = module.weight.data[mask == 0.0]
            max_leakage = pruned_weights.abs().max().item() if pruned_weights.numel() > 0 else 0.0
            assert max_leakage <= tolerance, (
                f"Mask integrity violated in layer {name}! "
                f"Pruned weights are non-zero (max absolute value: {max_leakage})"
            )


def prune_magnitude(
    model: nn.Module,
    current_masks: Dict[str, torch.Tensor],
    prune_rate: float = 0.20,
    mode: str = "global"
) -> Dict[str, torch.Tensor]:
    """
    Prunes the lowest-magnitude fraction `prune_rate` of the *currently remaining*
    weights using exact top-k indexing.

    Modes:
    - 'global': Exact bottom-k lowest magnitude weights across the entire network.
    - 'layerwise': Exact bottom-k lowest magnitude weights within each layer.
    """
    new_masks = {name: mask.clone() for name, mask in current_masks.items()}

    if mode == "global":
        # Build flattened views of all surviving weights with pointers back to layer and index
        layer_data = []
        all_surviving_weights = []

        for name, module in model.named_modules():
            if name in new_masks:
                mask_flat = new_masks[name].view(-1)
                weights_flat = module.weight.data.view(-1).abs().cpu()
                active_indices = (mask_flat == 1.0).nonzero(as_tuple=True)[0]
                active_weights = weights_flat[active_indices]

                layer_data.append((name, active_indices, len(active_weights)))
                all_surviving_weights.append(active_weights)

        if not all_surviving_weights:
            return new_masks

        concatenated_surviving = torch.cat(all_surviving_weights)
        k = int(round(prune_rate * concatenated_surviving.numel()))
        if k == 0:
            return new_masks

        # Find global bottom-k lowest magnitudes
        _, bottomk_indices = torch.topk(concatenated_surviving, k, largest=False)
        bottomk_set = set(bottomk_indices.tolist())

        # Disperse bottom-k back to respective layers
        offset = 0
        for name, active_indices, count in layer_data:
            # Indices belonging to this layer in concatenated vector
            layer_bottom_indices = [
                idx - offset for idx in bottomk_set if offset <= idx < offset + count
            ]
            if layer_bottom_indices:
                prune_at = active_indices[layer_bottom_indices]
                mask_flat = new_masks[name].view(-1)
                mask_flat[prune_at] = 0.0
                new_masks[name] = mask_flat.view_as(current_masks[name])
            offset += count

    elif mode == "layerwise":
        for name, module in model.named_modules():
            if name in new_masks:
                mask_flat = new_masks[name].view(-1)
                weights_flat = module.weight.data.view(-1).abs().cpu()
                active_indices = (mask_flat == 1.0).nonzero(as_tuple=True)[0]
                active_weights = weights_flat[active_indices]

                k = int(round(prune_rate * active_weights.numel()))
                if k > 0:
                    _, bottomk_idx = torch.topk(active_weights, k, largest=False)
                    prune_at = active_indices[bottomk_idx]
                    mask_flat[prune_at] = 0.0
                    new_masks[name] = mask_flat.view_as(current_masks[name])
    else:
        raise ValueError(f"Unknown pruning mode: {mode}. Choose 'global' or 'layerwise'.")

    return new_masks


def reset_to_initial_weights(
    model: nn.Module,
    initial_state_dict: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor]
) -> None:
    """
    Restores weights to θ0 and reapplies the mask.
    Non-prunable parameters (e.g. batch-norm running stats) are also restored to θ0.
    """
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    apply_masks_(model, masks)


def reinitialize_with_mask(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    reinit_seed: Optional[int] = None
) -> None:
    """
    Mandatory Control experiment:
    Re-initializes all weights from scratch with a new seed, but keeps
    the exact same binary pruning mask: θ_rand ⊙ m.
    """
    if reinit_seed is not None:
        set_seed(reinit_seed)

    # Call model's built-in initialization
    if hasattr(model, '_initialize_weights'):
        model._initialize_weights()
    else:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()

    # Reapply the mask to zero out pruned positions
    apply_masks_(model, masks)


def generate_random_mask(
    masks: Dict[str, torch.Tensor],
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Optional Extra Control:
    Generates a random binary mask that matches the exact layer-wise sparsity
    of the given mask, but with randomly shuffled zero/one positions.
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    random_masks = {}
    for name, mask in masks.items():
        total_elements = mask.numel()
        num_ones = int(mask.sum().item())
        flat_mask = torch.zeros(total_elements, dtype=torch.float32)
        if generator is not None:
            perm = torch.randperm(total_elements, generator=generator)
        else:
            perm = torch.randperm(total_elements)
        flat_mask[perm[:num_ones]] = 1.0
        random_masks[name] = flat_mask.view_as(mask)

    return random_masks
