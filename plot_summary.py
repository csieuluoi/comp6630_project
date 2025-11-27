import argparse
import json
import os

import matplotlib.pyplot as plt


def load_results(results_path: str):
    with open(results_path, "r") as f:
        data = json.load(f)
    return data


def plot_val_f1_by_trial(trials, out_path: str):
    trial_ids = [t["trial"] for t in trials]
    val_f1 = [t["val_macro_f1"] for t in trials]

    plt.figure(figsize=(8, 5))
    plt.plot(trial_ids, val_f1, marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Validation macro-F1")
    plt.title("Validation macro-F1 by Trial")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scalar_hparam_vs_f1(trials, hparam_name: str, out_path: str):
    """Scatter plot of a scalar hyperparameter vs validation F1."""
    x_vals = []
    y_vals = []

    for t in trials:
        hp = t["hparams"][hparam_name]
        # skip non-scalar hyperparams
        if isinstance(hp, (int, float)):
            x_vals.append(hp)
            y_vals.append(t["val_macro_f1"])

    if not x_vals:
        print(f"[WARN] No scalar values found for hyperparameter '{hparam_name}', skipping.")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, y_vals)
    plt.xlabel(hparam_name)
    plt.ylabel("Validation macro-F1")
    plt.title(f"Validation macro-F1 vs {hparam_name}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_categorical_hparam_box(trials, hparam_name: str, out_path: str):
    """Box/violin-style plot (simple box using matplotlib) for categorical hyperparams."""
    # Group F1 scores by hyperparameter value
    grouped = {}
    for t in trials:
        hp_val = t["hparams"][hparam_name]
        grouped.setdefault(str(hp_val), []).append(t["val_macro_f1"])

    labels = list(grouped.keys())
    data = [grouped[k] for k in labels]

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.boxplot(data, labels=labels)
    plt.xlabel(hparam_name)
    plt.ylabel("Validation macro-F1")
    plt.title(f"Validation macro-F1 by {hparam_name}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter search results for MLP cell type classifier"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/hparam_search_results.json",
        help="Path to hparam_search_results.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./figures",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results(args.results_path)
    trials = data["all_trials"]

    # 1) Validation F1 by trial index
    plot_val_f1_by_trial(
        trials, os.path.join(args.out_dir, "val_macro_f1_by_trial.png")
    )

    # 2) Scalar hyperparams vs F1
    scalar_hparams = ["learning_rate", "batch_size", "dropout", "weight_decay"]
    for hp in scalar_hparams:
        out_path = os.path.join(args.out_dir, f"val_macro_f1_vs_{hp}.png")
        plot_scalar_hparam_vs_f1(trials, hp, out_path)

    # 3) Categorical hyperparam: use_class_weights
    if any("use_class_weights" in t["hparams"] for t in trials):
        out_path = os.path.join(args.out_dir, "val_macro_f1_by_use_class_weights.png")
        plot_categorical_hparam_box(trials, "use_class_weights", out_path)

    # 4) Hidden dims treated as categorical (stringified)
    if any("hidden_dims" in t["hparams"] for t in trials):
        out_path = os.path.join(args.out_dir, "val_macro_f1_by_hidden_dims.png")
        plot_categorical_hparam_box(trials, "hidden_dims", out_path)

    print(f"Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
