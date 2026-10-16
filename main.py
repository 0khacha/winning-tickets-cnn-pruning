"""
Main Execution Script for Lottery Ticket Hypothesis Experiments.

Usage:
    python main.py --experiment sanity_check   # 1-epoch, 2-round rapid test of the full pipeline
    python main.py --experiment dense          # Runs dense baselines across seeds
    python main.py --experiment imp            # Runs full IMP Winning Tickets + Controls
    python main.py --experiment layerwise      # Runs Layer-wise IMP comparison
    python main.py --experiment transfer       # Runs CIFAR-10 -> SVHN transfer study
    python main.py --experiment plots          # Generates plots and tables from existing results
    python main.py --experiment all            # Runs all experiments sequentially
"""

import os
import argparse
import torch

from config import get_default_config, ExperimentConfig, set_seed
from experiments import (
    run_dense_baseline,
    run_imp_ticket_and_controls,
    run_transfer_study,
    load_json
)
from plots import (
    plot_accuracy_vs_sparsity,
    plot_learning_speed,
    plot_global_vs_layerwise,
    plot_transfer_learning,
    generate_results_table
)

def parse_args():
    parser = argparse.ArgumentParser(description="Lottery Ticket Hypothesis Runner")
    parser.add_argument(
        "--experiment", type=str, default="sanity_check",
        choices=["sanity_check", "dense", "imp", "layerwise", "transfer", "plots", "all"],
        help="Experiment to run"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per training round")
    parser.add_argument("--rounds", type=int, default=12, help="Number of IMP pruning rounds")
    parser.add_argument("--model", type=str, default="conv4", choices=["conv4", "resnet9"], help="Model architecture")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def get_sanity_dataloaders():
    """Lightweight synthetic dataset for rapid end-to-end pipeline verification (<15s on CPU)."""
    set_seed(42)
    x_train = torch.randn(32, 3, 32, 32)
    y_train = torch.randint(0, 10, (32,))
    x_val = torch.randn(16, 3, 32, 32)
    y_val = torch.randint(0, 10, (16,))
    x_test = torch.randn(16, 3, 32, 32)
    y_test = torch.randint(0, 10, (16,))
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    val_ds = torch.utils.data.TensorDataset(x_val, y_val)
    test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    return {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True),
        "val": torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False),
        "test": torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "num_classes": 10
    }


def run_sanity_check(cfg: ExperimentConfig):
    """
    Rapid end-to-end pipeline verification (< 1 minute on CPU):
    2 rounds, 1 epoch per round, 1 seed.
    Tests model initialization, pruning, mask enforcement,
    theta0 reset, random re-init, metrics logging, plotting, and table export.
    """
    print("\n" + "="*60)
    print(" >>> RUNNING QUICK SANITY CHECK (< 1 min end-to-end) <<<")
    print("="*60)

    cfg.seeds = [42]
    cfg.epochs = 1
    cfg.num_prune_rounds = 2
    cfg.output_dir = "./results/sanity_check"
    cfg.checkpoints_dir = "./results/sanity_check/checkpoints"
    cfg.plots_dir = "./results/sanity_check/plots"

    dl = get_sanity_dataloaders()

    # 1. Dense Baseline
    dense_res = run_dense_baseline(cfg, dataset_name=cfg.source_dataset, dataloaders=dl)

    # 2. IMP Winning Ticket and Controls
    imp_res = run_imp_ticket_and_controls(cfg, mode="global", include_random_mask_control=True, dataloaders=dl)

    # 3. Plots & Tables
    plot_accuracy_vs_sparsity(
        ticket_data=imp_res["ticket"],
        reinit_data=imp_res["reinit"],
        dense_data=dense_res,
        randmask_data=imp_res["randmask"],
        save_path=os.path.join(cfg.plots_dir, "sanity_accuracy_vs_sparsity.png")
    )
    plot_learning_speed(
        ticket_data=imp_res["ticket"],
        reinit_data=imp_res["reinit"],
        save_path=os.path.join(cfg.plots_dir, "sanity_learning_speed.png")
    )
    generate_results_table(
        ticket_data=imp_res["ticket"],
        reinit_data=imp_res["reinit"],
        randmask_data=imp_res["randmask"],
        save_path_md=os.path.join(cfg.output_dir, "sanity_results_table.md"),
        save_path_csv=os.path.join(cfg.output_dir, "sanity_results_table.csv")
    )

    print("\n" + "="*60)
    print(" >>> SANITY CHECK PASSED WITH ZERO ERRORS! <<<")
    print("="*60)


def main():
    args = parse_args()
    cfg = get_default_config()
    cfg.seeds = args.seeds
    cfg.epochs = args.epochs
    cfg.num_prune_rounds = args.rounds
    cfg.model_name = args.model
    cfg.device = args.device

    if args.experiment == "sanity_check":
        run_sanity_check(cfg)

    elif args.experiment == "dense":
        run_dense_baseline(cfg, dataset_name=cfg.source_dataset)
        run_dense_baseline(cfg, dataset_name=cfg.target_dataset)

    elif args.experiment == "imp":
        run_imp_ticket_and_controls(cfg, mode="global", include_random_mask_control=True)

    elif args.experiment == "layerwise":
        run_imp_ticket_and_controls(cfg, mode="layerwise", include_random_mask_control=False)

    elif args.experiment == "transfer":
        run_transfer_study(cfg, source_mode="global")

    elif args.experiment == "plots":
        # Load existing results from disk and generate all figures
        ticket_file = os.path.join(cfg.output_dir, f"imp_ticket_global_{cfg.model_name}.json")
        reinit_file = os.path.join(cfg.output_dir, f"imp_reinit_global_{cfg.model_name}.json")
        randmask_file = os.path.join(cfg.output_dir, f"imp_randmask_global_{cfg.model_name}.json")
        dense_file = os.path.join(cfg.output_dir, f"dense_baseline_{cfg.source_dataset}_{cfg.model_name}.json")

        tickets = load_json(ticket_file) or []
        reinits = load_json(reinit_file) or []
        randmasks = load_json(randmask_file) or []
        dense = load_json(dense_file) or []

        plot_accuracy_vs_sparsity(
            ticket_data=tickets,
            reinit_data=reinits,
            dense_data=dense,
            randmask_data=randmasks,
            save_path=os.path.join(cfg.plots_dir, "accuracy_vs_sparsity.png")
        )
        plot_learning_speed(
            ticket_data=tickets,
            reinit_data=reinits,
            save_path=os.path.join(cfg.plots_dir, "learning_speed.png")
        )
        generate_results_table(
            ticket_data=tickets,
            reinit_data=reinits,
            randmask_data=randmasks,
            save_path_md=os.path.join(cfg.output_dir, "results_table.md"),
            save_path_csv=os.path.join(cfg.output_dir, "results_table.csv")
        )

    elif args.experiment == "all":
        print("\nStarting full experiment suite...")
        dense_src = run_dense_baseline(cfg, dataset_name=cfg.source_dataset)
        dense_tgt = run_dense_baseline(cfg, dataset_name=cfg.target_dataset)
        glob_res = run_imp_ticket_and_controls(cfg, mode="global", include_random_mask_control=True)
        layer_res = run_imp_ticket_and_controls(cfg, mode="layerwise", include_random_mask_control=False)
        trans_res = run_transfer_study(cfg, source_mode="global")

        # Generate all plots
        plot_accuracy_vs_sparsity(
            ticket_data=glob_res["ticket"],
            reinit_data=glob_res["reinit"],
            dense_data=dense_src,
            randmask_data=glob_res["randmask"],
            save_path=os.path.join(cfg.plots_dir, "accuracy_vs_sparsity.png")
        )
        plot_learning_speed(
            ticket_data=glob_res["ticket"],
            reinit_data=glob_res["reinit"],
            save_path=os.path.join(cfg.plots_dir, "learning_speed.png")
        )
        plot_global_vs_layerwise(
            global_tickets=glob_res["ticket"],
            layerwise_tickets=layer_res["ticket"],
            save_path=os.path.join(cfg.plots_dir, "global_vs_layerwise.png")
        )
        plot_transfer_learning(
            transfer_data=trans_res["results"],
            target_dense_data=dense_tgt,
            save_path=os.path.join(cfg.plots_dir, "transfer_learning.png")
        )
        generate_results_table(
            ticket_data=glob_res["ticket"],
            reinit_data=glob_res["reinit"],
            randmask_data=glob_res["randmask"],
            save_path_md=os.path.join(cfg.output_dir, "results_table.md"),
            save_path_csv=os.path.join(cfg.output_dir, "results_table.csv")
        )


if __name__ == "__main__":
    main()
