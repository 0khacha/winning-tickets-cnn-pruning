"""
Experiment Orchestration Module for the Lottery Ticket Hypothesis.

Coordinates:
1. Dense Baseline (CIFAR-10 & SVHN) across multiple seeds.
2. Iterative Magnitude Pruning (IMP) with reset to θ0 (Winning Tickets).
3. Mandatory Control: Random Re-initialization with identical masks (θ_rand ⊙ m).
4. Extra Control: Random Masks with matched layer-wise sparsity.
5. Global vs Layer-wise IMP comparison.
6. Transfer Learning: CIFAR-10 tickets transferred to SVHN vs native tickets vs random masks.
7. Late Rewinding extension (θ_k vs θ_0 on ResNet-9).
8. Resumable checkpointing (CSV & JSON) and disk saving.
"""

import os
import json
from typing import Dict, List, Any, Optional
import torch

from config import ExperimentConfig, set_seed
from model import get_model
from data import get_dataloaders
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
from train import train_model

def save_json(data: Any, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Optional[Any]:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def run_dense_baseline(
    cfg: ExperimentConfig,
    dataset_name: Optional[str] = None,
    dataloaders: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Trains unpruned dense baselines across all configured random seeds.
    """
    dataset = dataset_name or cfg.source_dataset
    print(f"\n==========================================")
    print(f" RUNNING DENSE BASELINE ON {dataset.upper()}")
    print(f" Seeds: {cfg.seeds} | Epochs: {cfg.epochs}")
    print(f"==========================================")

    results_file = os.path.join(cfg.output_dir, f"dense_baseline_{dataset}_{cfg.model_name}.json")
    if cfg.resume:
        existing = load_json(results_file)
        if existing is not None:
            print(f"Loaded {len(existing)} existing dense baseline results from {results_file}")
            return existing

    results = []
    for seed in cfg.seeds:
        print(f"\n--- Training Dense Baseline (Seed {seed}) ---")
        set_seed(seed)

        dataloaders_dict = dataloaders if dataloaders is not None else get_dataloaders(
            dataset_name=dataset,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            val_split_ratio=cfg.val_split_ratio,
            seed=seed,
            num_workers=cfg.num_workers
        )

        model = get_model(cfg.model_name, in_channels=cfg.in_channels, num_classes=dataloaders_dict["num_classes"])
        masks = init_masks(model, prune_classifier=cfg.prune_classifier)

        train_res = train_model(
            model=model,
            dataloaders=dataloaders_dict,
            masks=masks,
            epochs=cfg.epochs,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            optimizer_type=cfg.optimizer,
            scheduler_type=cfg.lr_scheduler,
            device=cfg.device,
            verbose=True
        )

        sparsity_info = compute_actual_sparsity(masks)
        entry = {
            "experiment": "dense_baseline",
            "dataset": dataset,
            "model": cfg.model_name,
            "seed": seed,
            "round": 0,
            "fraction_surviving": sparsity_info["fraction_surviving"],
            "surviving_weights": sparsity_info["surviving_weights"],
            "total_weights": sparsity_info["total_weights"],
            "best_val_acc": train_res["best_val_acc"],
            "best_epoch": train_res["best_epoch"],
            "auc_val_acc": train_res["auc_val_acc"],
            "val_acc_epoch_5": train_res["val_acc_epoch_5"],
            "val_acc_epoch_10": train_res["val_acc_epoch_10"],
            "test_acc": train_res["test_acc"],
            "training_time": train_res["training_time"],
            "history": train_res["history"]
        }
        results.append(entry)
        save_json(results, results_file)
        print(f"Seed {seed} Dense Baseline: Val Acc = {train_res['best_val_acc']*100:.2f}%, Test Acc = {train_res['test_acc']*100:.2f}%")

    return results


def run_imp_ticket_and_controls(
    cfg: ExperimentConfig,
    mode: str = "global",
    include_random_mask_control: bool = True,
    dataloaders: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Executes Iterative Magnitude Pruning (IMP) and synchronous controls:
    1. Winning Ticket (IMP with reset to θ0 or θk if late rewinding)
    2. Mandatory Control: Random Re-init with same mask (θ_rand ⊙ m)
    3. Extra Control: Random mask with matched layer sparsity
    
    Fully supports crash resumption: saves and reloads masks, initial weights (θ0/θk),
    trained checkpoints, round state, and per-round metric records.
    """
    print(f"\n==========================================")
    print(f" RUNNING IMP EXPERIMENT (Mode: {mode.upper()})")
    print(f" Model: {cfg.model_name} | Dataset: {cfg.source_dataset}")
    print(f" Seeds: {cfg.seeds} | Rounds: {cfg.num_prune_rounds} | Prune Rate: {cfg.prune_rate*100}%")
    print(f"==========================================")

    res_dir = cfg.output_dir
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)

    ticket_file = os.path.join(res_dir, f"imp_ticket_{mode}_{cfg.model_name}.json")
    reinit_file = os.path.join(res_dir, f"imp_reinit_{mode}_{cfg.model_name}.json")
    randmask_file = os.path.join(res_dir, f"imp_randmask_{mode}_{cfg.model_name}.json")

    ticket_results = load_json(ticket_file) if cfg.resume else None
    ticket_results = ticket_results or []

    reinit_results = load_json(reinit_file) if cfg.resume else None
    reinit_results = reinit_results or []

    randmask_results = load_json(randmask_file) if cfg.resume else None
    randmask_results = randmask_results or []

    for seed in cfg.seeds:
        print(f"\n================== SEED {seed} ==================")
        set_seed(seed)

        dataloaders_dict = dataloaders if dataloaders is not None else get_dataloaders(
            dataset_name=cfg.source_dataset,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            val_split_ratio=cfg.val_split_ratio,
            seed=seed,
            num_workers=cfg.num_workers
        )

        # Initialize network and preserve or load θ0
        model = get_model(cfg.model_name, in_channels=cfg.in_channels, num_classes=dataloaders_dict["num_classes"])
        theta0_path = os.path.join(cfg.checkpoints_dir, f"theta0_{cfg.model_name}_seed_{seed}.pt")
        if cfg.resume and os.path.exists(theta0_path):
            theta0_sd = torch.load(theta0_path, map_location="cpu")
            model.load_state_dict(theta0_sd)
            print(f"Loaded existing theta0 initialization from {theta0_path}")
        else:
            theta0_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(theta0_sd, theta0_path)

        rewind_path = os.path.join(cfg.checkpoints_dir, f"thetak_{cfg.model_name}_seed_{seed}_epoch_{cfg.rewind_epoch}.pt") if cfg.rewind_epoch > 0 else None
        if rewind_path and os.path.exists(rewind_path):
            start_weights = torch.load(rewind_path, map_location="cpu")
        else:
            start_weights = theta0_sd

        for round_idx in range(cfg.num_prune_rounds + 1):
            mask_path = os.path.join(
                cfg.checkpoints_dir, f"mask_{mode}_{cfg.model_name}_seed_{seed}_round_{round_idx}.pt"
            )
            trained_ticket_path = os.path.join(
                cfg.checkpoints_dir, f"trained_ticket_{mode}_{cfg.model_name}_seed_{seed}_round_{round_idx}.pt"
            )

            # Retrieve or compute current_masks
            if cfg.resume and os.path.exists(mask_path):
                current_masks = torch.load(mask_path, map_location="cpu")
            elif round_idx == 0:
                current_masks = init_masks(model, prune_classifier=cfg.prune_classifier)
                torch.save(current_masks, mask_path)
            else:
                prev_mask_path = os.path.join(
                    cfg.checkpoints_dir, f"mask_{mode}_{cfg.model_name}_seed_{seed}_round_{round_idx - 1}.pt"
                )
                prev_ticket_path = os.path.join(
                    cfg.checkpoints_dir, f"trained_ticket_{mode}_{cfg.model_name}_seed_{seed}_round_{round_idx - 1}.pt"
                )
                if not os.path.exists(prev_ticket_path):
                    raise FileNotFoundError(f"Missing previous trained ticket checkpoint {prev_ticket_path} for round {round_idx}")
                prev_masks = torch.load(prev_mask_path, map_location="cpu")
                prev_sd = torch.load(prev_ticket_path, map_location="cpu")
                model.load_state_dict(prev_sd)
                current_masks = prune_magnitude(model=model, current_masks=prev_masks, prune_rate=cfg.prune_rate, mode=mode)
                torch.save(current_masks, mask_path)

            sparsity = compute_actual_sparsity(current_masks)
            frac_surviving = sparsity["fraction_surviving"]
            pct_surviving = frac_surviving * 100.0

            print(f"\n--- [Round {round_idx}/{cfg.num_prune_rounds}] Seed {seed} | Surviving Weights: {pct_surviving:.2f}% ---")

            # ---------------------------------------------------------
            # 1. WINNING TICKET (reset to start_weights, theta0 or thetak)
            # ---------------------------------------------------------
            ticket_entry = next((r for r in ticket_results if r["seed"] == seed and r["round"] == round_idx), None)
            ticket_done = (ticket_entry is not None) and os.path.exists(trained_ticket_path)
            if not ticket_done:
                print(f"Training Winning Ticket ({'thetak rewind' if cfg.rewind_epoch > 0 else 'theta0'} reset)...")
                reset_to_initial_weights(model, start_weights, current_masks)
                verify_mask_integrity(model, current_masks)

                t_res = train_model(
                    model=model,
                    dataloaders=dataloaders_dict,
                    masks=current_masks,
                    epochs=cfg.epochs,
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    optimizer_type=cfg.optimizer,
                    scheduler_type=cfg.lr_scheduler,
                    device=cfg.device,
                    verbose=False,
                    rewind_epoch=cfg.rewind_epoch
                )

                torch.save(t_res["best_model_weights"], trained_ticket_path)
                if round_idx == 0 and rewind_path and t_res.get("rewind_weights") is not None:
                    torch.save(t_res["rewind_weights"], rewind_path)
                    start_weights = t_res["rewind_weights"]

                ticket_entry = {
                    "experiment": "winning_ticket",
                    "mode": mode,
                    "model": cfg.model_name,
                    "seed": seed,
                    "round": round_idx,
                    "fraction_surviving": frac_surviving,
                    "surviving_weights": sparsity["surviving_weights"],
                    "total_weights": sparsity["total_weights"],
                    "best_val_acc": t_res["best_val_acc"],
                    "best_epoch": t_res["best_epoch"],
                    "auc_val_acc": t_res["auc_val_acc"],
                    "val_acc_epoch_5": t_res["val_acc_epoch_5"],
                    "val_acc_epoch_10": t_res["val_acc_epoch_10"],
                    "test_acc": t_res["test_acc"],
                    "training_time": t_res["training_time"],
                    "history": t_res["history"]
                }
                ticket_results = [r for r in ticket_results if not (r["seed"] == seed and r["round"] == round_idx)]
                ticket_results.append(ticket_entry)
                save_json(ticket_results, ticket_file)
                print(f"  -> Winning Ticket: Val Acc = {t_res['best_val_acc']*100:.2f}%, Test Acc = {t_res['test_acc']*100:.2f}% (Best Ep {t_res['best_epoch']})")
            else:
                print(f"  [Loaded Winning Ticket round {round_idx} from disk]")
                model.load_state_dict(torch.load(trained_ticket_path, map_location="cpu"))
                apply_masks_(model, current_masks)

            # ---------------------------------------------------------
            # 2. MANDATORY CONTROL: RANDOM RE-INITIALIZATION (θ_rand ⊙ m)
            # ---------------------------------------------------------
            if round_idx > 0:
                reinit_entry = next((r for r in reinit_results if r["seed"] == seed and r["round"] == round_idx), None)
                if reinit_entry is None:
                    print(f"Training Control: Random Re-init with same mask (theta_rand * m)...")
                    reinit_seed = seed * 1000 + round_idx
                    reinitialize_with_mask(model, current_masks, reinit_seed=reinit_seed)
                    verify_mask_integrity(model, current_masks)

                    r_res = train_model(
                        model=model,
                        dataloaders=dataloaders_dict,
                        masks=current_masks,
                        epochs=cfg.epochs,
                        lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay,
                        optimizer_type=cfg.optimizer,
                        scheduler_type=cfg.lr_scheduler,
                        device=cfg.device,
                        verbose=False
                    )

                    reinit_entry = {
                        "experiment": "random_reinit",
                        "mode": mode,
                        "model": cfg.model_name,
                        "seed": seed,
                        "round": round_idx,
                        "reinit_seed": reinit_seed,
                        "fraction_surviving": frac_surviving,
                        "surviving_weights": sparsity["surviving_weights"],
                        "total_weights": sparsity["total_weights"],
                        "best_val_acc": r_res["best_val_acc"],
                        "best_epoch": r_res["best_epoch"],
                        "auc_val_acc": r_res["auc_val_acc"],
                        "val_acc_epoch_5": r_res["val_acc_epoch_5"],
                        "val_acc_epoch_10": r_res["val_acc_epoch_10"],
                        "test_acc": r_res["test_acc"],
                        "training_time": r_res["training_time"],
                        "history": r_res["history"]
                    }
                    reinit_results.append(reinit_entry)
                    save_json(reinit_results, reinit_file)
                    print(f"  -> Random Re-init: Val Acc = {r_res['best_val_acc']*100:.2f}%, Test Acc = {r_res['test_acc']*100:.2f}%")
                else:
                    print(f"  [Loaded Random Re-init round {round_idx} from {reinit_file}]")

            # ---------------------------------------------------------
            # 3. EXTRA CONTROL: RANDOM MASK WITH MATCHED LAYER SPARSITY
            # ---------------------------------------------------------
            if round_idx > 0 and include_random_mask_control:
                randm_entry = next((r for r in randmask_results if r["seed"] == seed and r["round"] == round_idx), None)
                if randm_entry is None:
                    print(f"Training Extra Control: Random Mask (same layer sparsity)...")
                    rand_mask_seed = seed * 5000 + round_idx
                    random_m = generate_random_mask(current_masks, seed=rand_mask_seed)
                    reinitialize_with_mask(model, random_m, reinit_seed=rand_mask_seed)
                    verify_mask_integrity(model, random_m)

                    rm_res = train_model(
                        model=model,
                        dataloaders=dataloaders_dict,
                        masks=random_m,
                        epochs=cfg.epochs,
                        lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay,
                        optimizer_type=cfg.optimizer,
                        scheduler_type=cfg.lr_scheduler,
                        device=cfg.device,
                        verbose=False
                    )

                    randm_entry = {
                        "experiment": "random_mask",
                        "mode": mode,
                        "model": cfg.model_name,
                        "seed": seed,
                        "round": round_idx,
                        "randmask_seed": rand_mask_seed,
                        "fraction_surviving": frac_surviving,
                        "surviving_weights": sparsity["surviving_weights"],
                        "total_weights": sparsity["total_weights"],
                        "best_val_acc": rm_res["best_val_acc"],
                        "best_epoch": rm_res["best_epoch"],
                        "auc_val_acc": rm_res["auc_val_acc"],
                        "val_acc_epoch_5": rm_res["val_acc_epoch_5"],
                        "val_acc_epoch_10": rm_res["val_acc_epoch_10"],
                        "test_acc": rm_res["test_acc"],
                        "training_time": rm_res["training_time"],
                        "history": rm_res["history"]
                    }
                    randmask_results.append(randm_entry)
                    save_json(randmask_results, randmask_file)
                    print(f"  -> Random Mask:   Val Acc = {rm_res['best_val_acc']*100:.2f}%, Test Acc = {rm_res['test_acc']*100:.2f}%")
                else:
                    print(f"  [Loaded Random Mask round {round_idx} from {randmask_file}]")

            # Save checkpoint state record for crash recovery
            state_file = os.path.join(cfg.checkpoints_dir, f"state_{mode}_{cfg.model_name}_seed_{seed}.json")
            save_json({
                "seed": seed,
                "last_completed_round": round_idx,
                "num_prune_rounds": cfg.num_prune_rounds,
                "mode": mode,
                "model": cfg.model_name
            }, state_file)

            # Pre-compute and save next round's mask if not already saved
            if round_idx < cfg.num_prune_rounds:
                next_mask_path = os.path.join(
                    cfg.checkpoints_dir, f"mask_{mode}_{cfg.model_name}_seed_{seed}_round_{round_idx + 1}.pt"
                )
                if not os.path.exists(next_mask_path):
                    # Ensure model holds trained ticket parameters of current round
                    model.load_state_dict(torch.load(trained_ticket_path, map_location="cpu"))
                    next_mask = prune_magnitude(model=model, current_masks=current_masks, prune_rate=cfg.prune_rate, mode=mode)
                    torch.save(next_mask, next_mask_path)

    return {
        "ticket": ticket_results,
        "reinit": reinit_results,
        "randmask": randmask_results
    }


def run_transfer_study(
    cfg: ExperimentConfig,
    source_mode: str = "global"
) -> Dict[str, Any]:
    """
    Executes the Transfer Learning Study:
    Takes subnetwork masks found on CIFAR-10, resets feature extractor weights
    to θ0 (from CIFAR-10), re-initializes only the final classification layer,
    and trains on SVHN.

    Compares against:
    (a) Direct winning ticket found natively on SVHN.
    (b) Random mask with matched sparsity on SVHN.
    (c) Dense baseline on SVHN.
    """
    print(f"\n==========================================")
    print(f" RUNNING TRANSFER STUDY: CIFAR-10 -> SVHN")
    print(f" Seeds: {cfg.seeds} | Model: {cfg.model_name}")
    print(f"==========================================")

    res_file = os.path.join(cfg.output_dir, f"transfer_cifar10_to_{cfg.target_dataset}_{cfg.model_name}.json")
    if cfg.resume:
        existing = load_json(res_file)
        if existing is not None and len(existing) > 0:
            print(f"Loaded existing transfer study results from {res_file}")
            return {"results": existing}

    transfer_results = []

    for seed in cfg.seeds:
        print(f"\n--- Transfer Study (Seed {seed}) ---")
        set_seed(seed)

        # Target dataset dataloaders (SVHN)
        target_loaders = get_dataloaders(
            dataset_name=cfg.target_dataset,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            val_split_ratio=cfg.val_split_ratio,
            seed=seed,
            num_workers=cfg.num_workers
        )

        # Load CIFAR-10 initial weights θ0
        theta0_path = os.path.join(cfg.checkpoints_dir, f"theta0_{cfg.model_name}_seed_{seed}.pt")
        if not os.path.exists(theta0_path):
            raise FileNotFoundError(
                f"Source theta0 checkpoint not found at {theta0_path}! "
                f"Please run CIFAR-10 IMP first to generate source tickets."
            )
        theta0_sd = torch.load(theta0_path, map_location="cpu")

        # Select representative rounds to transfer (e.g. rounds 0, 3, 6, 9, 12)
        # to preserve compute budget while capturing the full sparsity spectrum
        rounds_to_transfer = [0, 3, 6, 9, cfg.num_prune_rounds]
        rounds_to_transfer = sorted(list(set(r for r in rounds_to_transfer if r <= cfg.num_prune_rounds)))

        for r in rounds_to_transfer:
            mask_path = os.path.join(
                cfg.checkpoints_dir, f"mask_{source_mode}_{cfg.model_name}_seed_{seed}_round_{r}.pt"
            )
            if not os.path.exists(mask_path):
                print(f"Mask {mask_path} not found. Skipping round {r}.")
                continue

            source_mask = torch.load(mask_path, map_location="cpu")
            sparsity = compute_actual_sparsity(source_mask)

            print(f"\n[Transfer Round {r}] Surviving: {sparsity['fraction_surviving']*100:.2f}%")

            # -----------------------------------------------------
            # A. Transferred Ticket (Mask & θ0 from CIFAR-10, new head for SVHN)
            # -----------------------------------------------------
            model_transfer = get_model(cfg.model_name, in_channels=cfg.in_channels, num_classes=10)
            reset_to_initial_weights(model_transfer, theta0_sd, source_mask)
            # Re-initialize the classifier head for SVHN
            model_transfer.reset_classifier(num_classes=target_loaders["num_classes"])
            apply_masks_(model_transfer, source_mask)
            verify_mask_integrity(model_transfer, source_mask)

            print(f"Training Transferred Ticket on {cfg.target_dataset}...")
            trans_res = train_model(
                model=model_transfer,
                dataloaders=target_loaders,
                masks=source_mask,
                epochs=cfg.epochs,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                optimizer_type=cfg.optimizer,
                scheduler_type=cfg.lr_scheduler,
                device=cfg.device,
                verbose=False
            )

            entry_trans = {
                "experiment": "transferred_ticket",
                "source_dataset": cfg.source_dataset,
                "target_dataset": cfg.target_dataset,
                "model": cfg.model_name,
                "seed": seed,
                "round": r,
                "fraction_surviving": sparsity["fraction_surviving"],
                "best_val_acc": trans_res["best_val_acc"],
                "best_epoch": trans_res["best_epoch"],
                "auc_val_acc": trans_res["auc_val_acc"],
                "test_acc": trans_res["test_acc"],
                "training_time": trans_res["training_time"]
            }
            transfer_results.append(entry_trans)

            # -----------------------------------------------------
            # B. Target Random Mask Control (Same sparsity on SVHN)
            # -----------------------------------------------------
            if r > 0:
                print(f"Training Random Mask Control on {cfg.target_dataset}...")
                rand_m = generate_random_mask(source_mask, seed=seed * 7000 + r)
                model_rand = get_model(cfg.model_name, in_channels=cfg.in_channels, num_classes=10)
                reinitialize_with_mask(model_rand, rand_m, reinit_seed=seed * 8000 + r)
                model_rand.reset_classifier(num_classes=target_loaders["num_classes"])
                apply_masks_(model_rand, rand_m)
                verify_mask_integrity(model_rand, rand_m)

                rm_res = train_model(
                    model=model_rand,
                    dataloaders=target_loaders,
                    masks=rand_m,
                    epochs=cfg.epochs,
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    optimizer_type=cfg.optimizer,
                    scheduler_type=cfg.lr_scheduler,
                    device=cfg.device,
                    verbose=False
                )

                entry_rand = {
                    "experiment": "target_random_mask",
                    "source_dataset": cfg.source_dataset,
                    "target_dataset": cfg.target_dataset,
                    "model": cfg.model_name,
                    "seed": seed,
                    "round": r,
                    "fraction_surviving": sparsity["fraction_surviving"],
                    "best_val_acc": rm_res["best_val_acc"],
                    "best_epoch": rm_res["best_epoch"],
                    "auc_val_acc": rm_res["auc_val_acc"],
                    "test_acc": rm_res["test_acc"],
                    "training_time": rm_res["training_time"]
                }
                transfer_results.append(entry_rand)

            save_json(transfer_results, res_file)
            print(f"  -> Transfer Ticket: Val Acc = {trans_res['best_val_acc']*100:.2f}%, Test Acc = {trans_res['test_acc']*100:.2f}%")

    return {"results": transfer_results}
