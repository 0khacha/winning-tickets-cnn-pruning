"""
Unit tests for prune.py, model.py, crash resumption, and determinism.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import get_default_config, set_seed
from model import get_model
from prune import (
    init_masks,
    apply_masks_,
    prune_magnitude,
    reset_to_initial_weights,
    reinitialize_with_mask,
    generate_random_mask,
    compute_actual_sparsity,
    verify_mask_integrity
)
from experiments import run_imp_ticket_and_controls

def test_pruning_flow():
    set_seed(42)
    model = get_model("conv4", in_channels=3, num_classes=10)
    initial_sd = {k: v.clone() for k, v in model.state_dict().items()}
    masks = init_masks(model, prune_classifier=False)

    # 1. Check initial sparsity is 100% surviving
    sp0 = compute_actual_sparsity(masks)
    assert sp0["fraction_surviving"] == 1.0, f"Expected 1.0, got {sp0['fraction_surviving']}"

    # 2. Global pruning round 1 (prune 20%)
    masks = prune_magnitude(model, masks, prune_rate=0.20, mode="global")
    sp1 = compute_actual_sparsity(masks)
    assert abs(sp1["fraction_surviving"] - 0.80) < 0.01, f"Expected ~0.80, got {sp1['fraction_surviving']}"

    # Apply masks and verify integrity
    apply_masks_(model, masks)
    verify_mask_integrity(model, masks)

    # 3. Reset to theta0
    reset_to_initial_weights(model, initial_sd, masks)
    verify_mask_integrity(model, masks)

    # Check that surviving weights match theta0
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name]
            theta0_w = initial_sd[f"{name}.weight"]
            assert torch.allclose(module.weight.data[mask == 1.0], theta0_w[mask == 1.0]), "Surviving weights must equal theta0"

    # 4. Mandatory Control: Random reinit with same mask
    reinitialize_with_mask(model, masks, reinit_seed=999)
    verify_mask_integrity(model, masks)
    sp_ctrl = compute_actual_sparsity(masks)
    assert sp_ctrl["fraction_surviving"] == sp1["fraction_surviving"]

    # 5. Extra Control: Random mask with same layer-wise sparsity
    rand_mask = generate_random_mask(masks, seed=888)
    for name in masks:
        assert masks[name].sum() == rand_mask[name].sum(), f"Layer {name} sum mismatch in random mask"

    # 6. Layerwise pruning round
    masks_lw = init_masks(model, prune_classifier=False)
    masks_lw = prune_magnitude(model, masks_lw, prune_rate=0.20, mode="layerwise")
    for name, mask in masks_lw.items():
        assert abs(mask.sum().item() / mask.numel() - 0.80) < 0.01, f"Layer {name} did not prune by 20%"

    print("ALL PRUNING UNIT TESTS PASSED SUCCESSFULLY!")

def get_synthetic_dataloaders():
    set_seed(123)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    return {
        "train": loader,
        "val": loader,
        "test": loader,
        "train_size": len(ds),
        "val_size": len(ds),
        "test_size": len(ds),
        "num_classes": 10
    }

def test_crash_resumption(tmp_path):
    dl = get_synthetic_dataloaders()

    # 1. Uninterrupted run: runs through round 3
    cfg_uninterrupted = get_default_config()
    cfg_uninterrupted.seeds = [42]
    cfg_uninterrupted.epochs = 1
    cfg_uninterrupted.num_prune_rounds = 3
    cfg_uninterrupted.output_dir = str(tmp_path / "uninterrupted")
    cfg_uninterrupted.checkpoints_dir = str(tmp_path / "uninterrupted" / "checkpoints")

    run_imp_ticket_and_controls(
        cfg_uninterrupted, mode="global", include_random_mask_control=True, dataloaders=dl
    )
    uninterrupted_mask_r3 = torch.load(
        os.path.join(cfg_uninterrupted.checkpoints_dir, "mask_global_conv4_seed_42_round_3.pt"),
        map_location="cpu"
    )

    # 2. Interrupted run: stops after round 2
    cfg_interrupted = get_default_config()
    cfg_interrupted.seeds = [42]
    cfg_interrupted.epochs = 1
    cfg_interrupted.num_prune_rounds = 2
    cfg_interrupted.output_dir = str(tmp_path / "resumed")
    cfg_interrupted.checkpoints_dir = str(tmp_path / "resumed" / "checkpoints")

    run_imp_ticket_and_controls(
        cfg_interrupted, mode="global", include_random_mask_control=True, dataloaders=dl
    )

    # Verify that initial weights, trained weights, and masks were saved for round 2
    assert os.path.exists(os.path.join(cfg_interrupted.checkpoints_dir, "theta0_conv4_seed_42.pt"))
    assert os.path.exists(os.path.join(cfg_interrupted.checkpoints_dir, "trained_ticket_global_conv4_seed_42_round_2.pt"))
    assert os.path.exists(os.path.join(cfg_interrupted.checkpoints_dir, "mask_global_conv4_seed_42_round_2.pt"))

    # 3. Resume the interrupted run to round 3
    cfg_interrupted.num_prune_rounds = 3
    cfg_interrupted.resume = True

    run_imp_ticket_and_controls(
        cfg_interrupted, mode="global", include_random_mask_control=True, dataloaders=dl
    )
    resumed_mask_r3 = torch.load(
        os.path.join(cfg_interrupted.checkpoints_dir, "mask_global_conv4_seed_42_round_3.pt"),
        map_location="cpu"
    )

    # 4. Check that masks after round 3 are exactly identical
    for name in uninterrupted_mask_r3:
        assert torch.equal(uninterrupted_mask_r3[name], resumed_mask_r3[name]), (
            f"Mask mismatch in layer {name} between uninterrupted and resumed runs!"
        )

def test_seed_determinism():
    """Verify that running with the same seed twice produces bitwise identical weights and loss."""
    def run_one():
        set_seed(1234)
        model = get_model("conv4")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(64, 3, 32, 32)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(x, y), batch_size=32)
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            out = model(bx)
            loss = F.cross_entropy(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        return total_loss, {k: v.clone() for k, v in model.state_dict().items()}

    loss1, sd1 = run_one()
    loss2, sd2 = run_one()

    assert abs(loss1 - loss2) < 1e-7, f"Loss mismatch: {loss1} vs {loss2}"
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"Weight mismatch in parameter {k}"

if __name__ == "__main__":
    test_pruning_flow()
    test_seed_determinism()
