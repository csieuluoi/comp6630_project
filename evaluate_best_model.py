import os
import argparse
import json
import shutil

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import scanpy as sc
import anndata as ad

from model import MLP, MLPConfig
from train import load_data, get_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate best MLP model on validation and test sets and plot UMAPs"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU id to use (same semantics as in train.py)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory where best_model.pth and hparam_search_results.json are stored",
    )
    return parser.parse_args()


def set_device(gpu_id: int) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


@torch.no_grad()
def get_predictions_and_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """
    Returns:
        all_labels: (N,) int array of true labels
        all_preds:  (N,) int array of predicted labels
        all_embs:   (N, D) float array of embeddings (here: logits)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_embs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # logits
        _, preds = torch.max(outputs, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_embs.append(outputs.cpu().numpy())  # use logits as embeddings

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_embs = np.concatenate(all_embs)

    return all_labels, all_preds, all_embs


def build_label_encoder_from_mapping(label_mapping: dict) -> LabelEncoder:
    """
    label_mapping is {int_index: class_name} as saved in the hparam search script,
    but after json.load the keys will be strings. We restore the original order.
    """
    indices_classes = sorted(
        [(int(k), v) for k, v in label_mapping.items()], key=lambda x: x[0]
    )
    classes = [cls for _, cls in indices_classes]

    le = LabelEncoder()
    le.classes_ = np.array(classes, dtype=object)
    return le


def save_classification_report(
    y_true,
    y_pred,
    le: LabelEncoder,
    split_name: str,
    results_dir: str,
):
    report = classification_report(
        y_true,
        y_pred,
        target_names=le.classes_,
        digits=4,
    )

    report_path = os.path.join(results_dir, f"{split_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{split_name.capitalize()} set classification report\n")
        f.write("=" * 80 + "\n")
        f.write(report + "\n")

    print(f"Saved {split_name} classification report to {report_path}")


def plot_umap(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    split_name: str,
    results_dir: str,
):
    """
    Save embeddings as an AnnData object and plot UMAP colored by true and predicted labels.
    """
    # Reconstruct string labels
    true_labels_str = le.inverse_transform(y_true)
    pred_labels_str = le.inverse_transform(y_pred)

    # Create AnnData
    adata = ad.AnnData(X=embeddings)
    adata.obs["label"] = true_labels_str.astype(str)
    adata.obs["predicted_label"] = pred_labels_str.astype(str)


    # Configure Scanpy output directory
    sc.settings.figdir = results_dir

    # Compute neighbors & UMAP, then plot
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=["label", "predicted_label"],
        legend_loc="on data",
        save=f"_{split_name}_embeddings.png",
        show=False,
    )
    print(
        f"Saved {split_name} UMAP plot to "
        f"{os.path.join(results_dir, f'figures/umap_{split_name}_embeddings.png')}"
    )
    ## move the figure to "./figures"
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    src_path = os.path.join(results_dir, f"figures/umap_{split_name}_embeddings.png")
    dst_path = os.path.join(figures_dir, f"umap_{split_name}_embeddings.png")
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)


def main():
    args = parse_args()
    device = set_device(args.gpu)

    os.makedirs(args.results_dir, exist_ok=True)

    # Load hyperparameter search results and best model weights
    results_path = os.path.join(args.results_dir, "hparam_search_results.json")
    best_model_path = os.path.join(args.results_dir, "best_model.pth")

    with open(results_path, "r") as f:
        results = json.load(f)

    best_hparams = results["best_hyperparameters"]
    label_mapping = results["label_mapping"]

    # Rebuild LabelEncoder to match the one used in training
    le = build_label_encoder_from_mapping(label_mapping)

    # Load data and encode labels using the same mapping
    train_X, train_y_raw, val_X, val_y_raw, test_X, test_y_raw = load_data()

    train_y = le.transform(train_y_raw)
    val_y = le.transform(val_y_raw)
    test_y = le.transform(test_y_raw)

    input_dim = train_X.shape[1]
    num_classes = len(le.classes_)

    # Rebuild the best model
    best_model_config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=best_hparams["hidden_dims"],
        output_dim=num_classes,
        dropout_rate=best_hparams["dropout"],
    )

    model = MLP(best_model_config).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    # DataLoaders for validation and test sets
    batch_size = best_hparams["batch_size"]

    val_loader = get_loader(val_X, val_y, batch_size=batch_size, shuffle=False)
    test_loader = get_loader(test_X, test_y, batch_size=batch_size, shuffle=False)

    # Get predictions & embeddings
    val_true, val_pred, val_embs = get_predictions_and_embeddings(
        model, val_loader, device
    )
    test_true, test_pred, test_embs = get_predictions_and_embeddings(
        model, test_loader, device
    )

    # Save classification reports
    save_classification_report(val_true, val_pred, le, "val", args.results_dir)
    save_classification_report(test_true, test_pred, le, "test", args.results_dir)

    # Save embeddings + UMAP plots using Scanpy
    plot_umap(
        val_embs, val_true, val_pred, le, "val", args.results_dir
    )
    plot_umap(
        test_embs, test_true, test_pred, le, "test", args.results_dir
    )


if __name__ == "__main__":
    main()
