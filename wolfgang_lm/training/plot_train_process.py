import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training metrics for Wolfgang-LM"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        default="finetune",
        help="Mode to plot: 'pretrain' or 'finetune'",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override output directory (default: out-pretrain or out-finetune based on mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration based on mode
    if args.mode == "pretrain":
        default_dir = "out-pretrain"
        train_filename = "train_log.csv"
        val_filename = "val_log.csv"
        output_prefix = "pretrain"
        title_prefix = "Pre-training"
    else:
        default_dir = "out-finetune"
        train_filename = "train_finetune_log.csv"
        val_filename = "val_finetune_log.csv"
        output_prefix = "finetune"
        title_prefix = "Fine-tuning"

    # Use user provided dir or default
    log_dir = args.out_dir if args.out_dir else default_dir

    train_path = os.path.join(log_dir, train_filename)
    val_path = os.path.join(log_dir, val_filename)

    print(f"Loading logs from {log_dir}...")

    # Load data
    try:
        train_log = pd.read_csv(train_path)
        val_log = pd.read_csv(val_path)
    except FileNotFoundError as e:
        print(f"Error: Log file not found: {e.filename}")
        print(f"searched in: {log_dir}")
        sys.exit(1)

    print("--- Analysis Report ---")
    print(f"Total Training Steps logged: {len(train_log)}")
    if not train_log.empty:
        print(f"Final Training Loss: {train_log['loss'].iloc[-1]:.4f}")
        print(f"Final Training PPL: {train_log['ppl'].iloc[-1]:.4f}")
        print(f"Max Gradient Norm: {train_log['grad_norm'].max():.4f}")

    if not val_log.empty:
        min_val_loss_idx = val_log["val_loss"].idxmin()
        min_val_loss_row = val_log.loc[min_val_loss_idx]
        print(
            f"Minimum Validation Loss: {min_val_loss_row['val_loss']:.4f} at iter {int(min_val_loss_row['iter'])}"
        )
        print(f"Minimum Validation PPL: {min_val_loss_row['val_ppl']:.4f}")

        # Check for overfitting (val loss increasing while train loss decreasing)
        # Simple check: compare last val loss with min val loss
        last_val_row = val_log.iloc[-1]
        if last_val_row["val_loss"] > min_val_loss_row["val_loss"]:
            print(
                f"Warning: Potential Overfitting. Last Val Loss ({last_val_row['val_loss']:.4f}) > Min Val Loss."
            )

    # Create plots
    plt.style.use("ggplot")
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f"{title_prefix} Metrics Analysis", fontsize=16)

    # 1. Loss (Train vs Val)
    axes[0, 0].plot(train_log["iter"], train_log["loss"], label="Train Loss", alpha=0.7)
    # Use exact iterations for validation from the logs
    axes[0, 0].plot(
        val_log["iter"], val_log["val_loss"], label="Val Loss", marker="o", linewidth=2
    )
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Perplexity (Train vs Val)
    axes[0, 1].plot(train_log["iter"], train_log["ppl"], label="Train PPL", alpha=0.7)
    axes[0, 1].plot(
        val_log["iter"], val_log["val_ppl"], label="Val PPL", marker="o", linewidth=2
    )
    axes[0, 1].set_title("Perplexity (Log Scale)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("PPL")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Learning Rate
    axes[1, 0].plot(train_log["iter"], train_log["lr"], color="tab:green")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].grid(True)

    # 4. Gradient Norm
    axes[1, 1].plot(train_log["iter"], train_log["grad_norm"], color="tab:purple")
    axes[1, 1].set_title("Gradient Norm")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Norm")
    axes[1, 1].grid(True)

    # 5. Training Speed (Time per iter)
    axes[2, 0].plot(train_log["iter"], train_log["time_ms"], color="tab:orange")
    axes[2, 0].set_title("Time per Iteration")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("Time (ms)")
    axes[2, 0].grid(True)

    # Remove empty subplot
    fig.delaxes(axes[2, 1])

    plt.tight_layout()
    output_file = os.path.join(log_dir, f"{output_prefix}_metrics.png")
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")


if __name__ == "__main__":
    main()
