"""
Visualization and Table Generation for Lottery Ticket Experiments.

Generates publication-quality plots:
1. Accuracy vs % Weights Remaining (Log-scale x-axis with mean ± std error bands).
2. Learning Speed Dynamics (Epoch to best accuracy & AUC-val vs Sparsity).
3. Global vs Layer-wise Pruning comparison.
4. Transfer Learning Curves (CIFAR-10 -> SVHN vs controls).
5. Comprehensive Results Summary Table (Markdown, CSV & LaTeX).
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'figure.autolayout': True
})


def aggregate_runs(records: List[Dict[str, Any]], metric_key: str = "test_acc") -> pd.DataFrame:
    """Aggregates multiple seed runs into mean and standard deviation per round."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["round", "fraction_surviving"])[metric_key].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["pct_surviving"] = grouped["fraction_surviving"] * 100.0
    return grouped.sort_values("pct_surviving", ascending=False)


def plot_accuracy_vs_sparsity(
    ticket_data: List[Dict[str, Any]],
    reinit_data: List[Dict[str, Any]],
    dense_data: Optional[List[Dict[str, Any]]] = None,
    randmask_data: Optional[List[Dict[str, Any]]] = None,
    save_path: str = "./results/plots/accuracy_vs_sparsity.png",
    metric: str = "test_acc",
    title_suffix: str = "(Conv-4 on CIFAR-10)"
) -> None:
    """
    Core Lottery Ticket plot:
    Accuracy vs % Weights Remaining on a log-scale x-axis with error bands.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    df_ticket = aggregate_runs(ticket_data, metric_key=metric)
    df_reinit = aggregate_runs(reinit_data, metric_key=metric)

    # 1. Winning Ticket Curve
    if not df_ticket.empty:
        ax.plot(
            df_ticket["pct_surviving"], df_ticket["mean"] * 100.0,
            marker='o', color='#1f77b4', label=r'Winning Ticket (Reset to $\theta_0$)'
        )
        ax.fill_between(
            df_ticket["pct_surviving"],
            (df_ticket["mean"] - df_ticket["std"]) * 100.0,
            (df_ticket["mean"] + df_ticket["std"]) * 100.0,
            color='#1f77b4', alpha=0.2
        )

    # 2. Random Re-init Control Curve
    if not df_reinit.empty:
        ax.plot(
            df_reinit["pct_surviving"], df_reinit["mean"] * 100.0,
            marker='s', color='#d62728', linestyle='--', label=r'Random Re-init Control ($\theta_{rand} \odot m$)'
        )
        ax.fill_between(
            df_reinit["pct_surviving"],
            (df_reinit["mean"] - df_reinit["std"]) * 100.0,
            (df_reinit["mean"] + df_reinit["std"]) * 100.0,
            color='#d62728', alpha=0.15
        )

    # 3. Optional Extra Control: Random Mask
    if randmask_data is not None:
        df_rm = aggregate_runs(randmask_data, metric_key=metric)
        if not df_rm.empty:
            ax.plot(
                df_rm["pct_surviving"], df_rm["mean"] * 100.0,
                marker='^', color='#ff7f0e', linestyle=':', label='Random Mask Control'
            )
            ax.fill_between(
                df_rm["pct_surviving"],
                (df_rm["mean"] - df_rm["std"]) * 100.0,
                (df_rm["mean"] + df_rm["std"]) * 100.0,
                color='#ff7f0e', alpha=0.15
            )

    # 4. Dense Baseline horizontal line
    if dense_data is not None and len(dense_data) > 0:
        dense_vals = [d[metric] * 100.0 for d in dense_data]
        dense_mean = np.mean(dense_vals)
        dense_std = np.std(dense_vals)
        ax.axhline(dense_mean, color='#2ca02c', linestyle='-.', label=f'Dense Baseline ({dense_mean:.1f}%)')
        ax.axhspan(dense_mean - dense_std, dense_mean + dense_std, color='#2ca02c', alpha=0.1)

    # Formatting x-axis as inverted log-scale
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([100, 50, 20, 10, 5, 2])

    metric_name = "Test Accuracy (%)" if metric == "test_acc" else "Validation Accuracy (%)"
    ax.set_xlabel("Weights Remaining (%) — Log Scale (decreasing →)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"The Lottery Ticket Hypothesis: {metric_name} vs Sparsity\n{title_suffix}", fontweight='bold')
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs sparsity plot to: {save_path}")


def plot_learning_speed(
    ticket_data: List[Dict[str, Any]],
    reinit_data: List[Dict[str, Any]],
    save_path: str = "./results/plots/learning_speed.png"
) -> None:
    """
    Plots learning speed metrics:
    - Left: Epoch at which best validation accuracy is achieved vs % Weights Remaining.
    - Right: Area Under the Validation Curve (AUC-val) vs % Weights Remaining.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Early-training accuracy (Epoch 5)
    # Check if val_acc_epoch_5 exists, fallback to best_val_acc
    metric_early = "val_acc_epoch_5" if any("val_acc_epoch_5" in d for d in ticket_data) else "best_val_acc"
    df_ep_ticket = aggregate_runs(ticket_data, metric_key=metric_early)
    df_ep_reinit = aggregate_runs(reinit_data, metric_key=metric_early)

    if not df_ep_ticket.empty:
        ax1.plot(df_ep_ticket["pct_surviving"], df_ep_ticket["mean"] * 100.0, marker='o', color='#1f77b4', label='Winning Ticket')
        ax1.fill_between(
            df_ep_ticket["pct_surviving"],
            (df_ep_ticket["mean"] - df_ep_ticket["std"]) * 100.0,
            (df_ep_ticket["mean"] + df_ep_ticket["std"]) * 100.0,
            color='#1f77b4', alpha=0.2
        )
    if not df_ep_reinit.empty:
        ax1.plot(df_ep_reinit["pct_surviving"], df_ep_reinit["mean"] * 100.0, marker='s', color='#d62728', linestyle='--', label='Random Re-init')
        ax1.fill_between(
            df_ep_reinit["pct_surviving"],
            (df_ep_reinit["mean"] - df_ep_reinit["std"]) * 100.0,
            (df_ep_reinit["mean"] + df_ep_reinit["std"]) * 100.0,
            color='#d62728', alpha=0.15
        )

    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks([100, 50, 20, 10, 5, 2])
    ax1.set_xlabel("Weights Remaining (%) — Log Scale")
    ax1.set_ylabel("Early Validation Accuracy (%) [Epoch 5]")
    ax1.set_title("Early Convergence Speed (Epoch 5 Val Acc)", fontweight='bold')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.4)

    # AUC of Validation Accuracy
    df_auc_ticket = aggregate_runs(ticket_data, metric_key="auc_val_acc")
    df_auc_reinit = aggregate_runs(reinit_data, metric_key="auc_val_acc")

    if not df_auc_ticket.empty:
        ax2.plot(df_auc_ticket["pct_surviving"], df_auc_ticket["mean"] * 100.0, marker='o', color='#1f77b4', label='Winning Ticket')
        ax2.fill_between(
            df_auc_ticket["pct_surviving"],
            (df_auc_ticket["mean"] - df_auc_ticket["std"]) * 100.0,
            (df_auc_ticket["mean"] + df_auc_ticket["std"]) * 100.0,
            color='#1f77b4', alpha=0.2
        )
    if not df_auc_reinit.empty:
        ax2.plot(df_auc_reinit["pct_surviving"], df_auc_reinit["mean"] * 100.0, marker='s', color='#d62728', linestyle='--', label='Random Re-init')
        ax2.fill_between(
            df_auc_reinit["pct_surviving"],
            (df_auc_reinit["mean"] - df_auc_reinit["std"]) * 100.0,
            (df_auc_reinit["mean"] + df_auc_reinit["std"]) * 100.0,
            color='#d62728', alpha=0.15
        )

    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks([100, 50, 20, 10, 5, 2])
    ax2.set_xlabel("Weights Remaining (%) — Log Scale")
    ax2.set_ylabel("Area Under Validation Curve (AUC-val %)")
    ax2.set_title("Training Efficiency: AUC of Validation Accuracy", fontweight='bold')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved learning speed plot to: {save_path}")


def plot_global_vs_layerwise(
    global_tickets: List[Dict[str, Any]],
    layerwise_tickets: List[Dict[str, Any]],
    save_path: str = "./results/plots/global_vs_layerwise.png"
) -> None:
    """Compares Global IMP against Layer-wise IMP."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    df_glob = aggregate_runs(global_tickets, metric_key="test_acc")
    df_layer = aggregate_runs(layerwise_tickets, metric_key="test_acc")

    if not df_glob.empty:
        ax.plot(df_glob["pct_surviving"], df_glob["mean"] * 100.0, marker='o', color='#1f77b4', label='Global IMP Ticket')
        ax.fill_between(
            df_glob["pct_surviving"],
            (df_glob["mean"] - df_glob["std"]) * 100.0,
            (df_glob["mean"] + df_glob["std"]) * 100.0,
            color='#1f77b4', alpha=0.2
        )

    if not df_layer.empty:
        ax.plot(df_layer["pct_surviving"], df_layer["mean"] * 100.0, marker='^', color='#9467bd', linestyle='--', label='Layer-wise IMP Ticket')
        ax.fill_between(
            df_layer["pct_surviving"],
            (df_layer["mean"] - df_layer["std"]) * 100.0,
            (df_layer["mean"] + df_layer["std"]) * 100.0,
            color='#9467bd', alpha=0.2
        )

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([100, 50, 20, 10, 5, 2])
    ax.set_xlabel("Weights Remaining (%) — Log Scale")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Global vs Layer-wise Iterative Magnitude Pruning", fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved global vs layer-wise plot to: {save_path}")


def plot_transfer_learning(
    transfer_data: List[Dict[str, Any]],
    target_dense_data: Optional[List[Dict[str, Any]]] = None,
    save_path: str = "./results/plots/transfer_learning.png"
) -> None:
    """Plots transfer learning performance of CIFAR-10 tickets evaluated on SVHN."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    df_all = pd.DataFrame(transfer_data)
    if df_all.empty:
        return

    df_trans = df_all[df_all["experiment"] == "transferred_ticket"]
    df_rand = df_all[df_all["experiment"] == "target_random_mask"]

    if not df_trans.empty:
        grp_trans = df_trans.groupby(["round", "fraction_surviving"])["test_acc"].agg(["mean", "std"]).reset_index()
        grp_trans["pct_surviving"] = grp_trans["fraction_surviving"] * 100.0
        ax.plot(
            grp_trans["pct_surviving"], grp_trans["mean"] * 100.0,
            marker='o', color='#2ca02c', label='Transferred Ticket (CIFAR-10 → SVHN)'
        )
        ax.fill_between(
            grp_trans["pct_surviving"],
            (grp_trans["mean"] - grp_trans["std"]) * 100.0,
            (grp_trans["mean"] + grp_trans["std"]) * 100.0,
            color='#2ca02c', alpha=0.2
        )

    if not df_rand.empty:
        grp_rand = df_rand.groupby(["round", "fraction_surviving"])["test_acc"].agg(["mean", "std"]).reset_index()
        grp_rand["pct_surviving"] = grp_rand["fraction_surviving"] * 100.0
        ax.plot(
            grp_rand["pct_surviving"], grp_rand["mean"] * 100.0,
            marker='x', color='#7f7f7f', linestyle='--', label='Random Mask on SVHN'
        )
        ax.fill_between(
            grp_rand["pct_surviving"],
            (grp_rand["mean"] - grp_rand["std"]) * 100.0,
            (grp_rand["mean"] + grp_rand["std"]) * 100.0,
            color='#7f7f7f', alpha=0.15
        )

    if target_dense_data is not None and len(target_dense_data) > 0:
        dense_vals = [d["test_acc"] * 100.0 for d in target_dense_data]
        dense_mean = np.mean(dense_vals)
        ax.axhline(dense_mean, color='#17becf', linestyle='-.', label=f'Dense SVHN Baseline ({dense_mean:.1f}%)')

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([100, 50, 20, 10, 5, 2])
    ax.set_xlabel("Weights Remaining (%) — Log Scale")
    ax.set_ylabel("SVHN Test Accuracy (%)")
    ax.set_title("Subnetwork Transferability: CIFAR-10 Ticket on SVHN", fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved transfer learning plot to: {save_path}")


def generate_results_table(
    ticket_data: List[Dict[str, Any]],
    reinit_data: List[Dict[str, Any]],
    randmask_data: Optional[List[Dict[str, Any]]] = None,
    save_path_md: str = "./results/results_table.md",
    save_path_csv: str = "./results/results_table.csv"
) -> pd.DataFrame:
    """
    Builds a structured results table comparing Ticket vs Random Reinit vs Random Mask across rounds.
    """
    df_t = aggregate_runs(ticket_data, metric_key="test_acc")
    df_r = aggregate_runs(reinit_data, metric_key="test_acc")
    df_rm = aggregate_runs(randmask_data, metric_key="test_acc") if randmask_data else None

    # Merge by round
    merged = pd.merge(
        df_t[["round", "pct_surviving", "mean", "std"]],
        df_r[["round", "mean", "std"]],
        on="round",
        how="left",
        suffixes=("_ticket", "_reinit")
    )

    merged["Ticket Test Acc (%)"] = merged.apply(
        lambda row: f"{row['mean_ticket']*100:.2f} ± {row['std_ticket']*100:.2f}", axis=1
    )
    merged["Reinit Test Acc (%)"] = merged.apply(
        lambda row: f"{row['mean_reinit']*100:.2f} ± {row['std_reinit']*100:.2f}" if pd.notnull(row['mean_reinit']) else "—", axis=1
    )
    merged["Delta (Ticket - Reinit)"] = merged.apply(
        lambda row: f"{(row['mean_ticket'] - row['mean_reinit'])*100:+.2f}%" if pd.notnull(row['mean_reinit']) else "—", axis=1
    )

    if df_rm is not None and not df_rm.empty:
        merged = pd.merge(merged, df_rm[["round", "mean", "std"]], on="round", how="left")
        merged["RandMask Test Acc (%)"] = merged.apply(
            lambda row: f"{row['mean']*100:.2f} ± {row['std']*100:.2f}" if pd.notnull(row['mean']) else "—", axis=1
        )
        display_cols = [
            "round", "pct_surviving", "Ticket Test Acc (%)",
            "Reinit Test Acc (%)", "Delta (Ticket - Reinit)", "RandMask Test Acc (%)"
        ]
    else:
        display_cols = [
            "round", "pct_surviving", "Ticket Test Acc (%)",
            "Reinit Test Acc (%)", "Delta (Ticket - Reinit)"
        ]

    final_table = merged[display_cols].copy()
    final_table.rename(columns={"round": "Round", "pct_surviving": "Weights Remaining (%)"}, inplace=True)
    final_table["Weights Remaining (%)"] = final_table["Weights Remaining (%)"].map(lambda x: f"{x:.2f}%")

    os.makedirs(os.path.dirname(save_path_md), exist_ok=True)
    with open(save_path_md, "w") as f:
        f.write("# Lottery Ticket Hypothesis: Experimental Results Summary\n\n")
        f.write(final_table.to_markdown(index=False))
        f.write("\n")

    final_table.to_csv(save_path_csv, index=False)
    print(f"Results summary saved to:\n  - Markdown: {save_path_md}\n  - CSV: {save_path_csv}")
    return final_table
