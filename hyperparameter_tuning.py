import os
import argparse
import copy
import json
import random

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from model import MLP, MLPConfig 
from train import load_data, get_loader, train_loop  

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for MLP cell type classifier"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use (same semantics as in train.py)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of random hyperparameter trials to run",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs per trial (early stopping is inside train_loop)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:

    class_counts = np.bincount(y)
    num_classes = len(class_counts)
    total = len(y)

    # avoid division by zero (shouldn't happen if all classes appear in train)
    class_counts = np.maximum(class_counts, 1)

    weights = total / (num_classes * class_counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_macro_f1(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: torch.device) -> float:
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return f1_score(all_labels, all_preds, average="macro")


def sample_hyperparams():
    hidden_dim_candidates = [
        [128, 64],
        [256, 128],
        [512, 256],
        [512, 256, 128],
    ]

    learning_rates = [1e-3, 3e-4, 1e-4]
    batch_sizes = [128, 256, 512]
    dropouts = [0.3, 0.5]
    weight_decays = [0.0, 1e-4, 1e-3]
    use_class_weights = [False, True]

    return {
        "hidden_dims": random.choice(hidden_dim_candidates),
        "learning_rate": random.choice(learning_rates),
        "batch_size": random.choice(batch_sizes),
        "dropout": random.choice(dropouts),
        "weight_decay": random.choice(weight_decays),
        "use_class_weights": random.choice(use_class_weights),
    }


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    os.makedirs("./results", exist_ok=True)

    # Load data
    train_X, train_y, val_X, val_y, test_X, test_y = load_data()

    le = LabelEncoder()
    train_y = le.fit_transform(train_y)
    val_y = le.transform(val_y)
    test_y = le.transform(test_y)

    input_dim = train_X.shape[1]
    num_classes = len(le.classes_)


    best_val_macro_f1 = -1.0
    best_state_dict = None
    best_hparams = None

    all_results = []

    print(f"Starting hyperparameter search with {args.n_trials} trials...\n")

    for trial in range(1, args.n_trials + 1):
        hparams = sample_hyperparams()

        print("=" * 80)
        print(f"Trial {trial}/{args.n_trials}")
        print("Hyperparameters:")
        for k, v in hparams.items():
            print(f"  {k}: {v}")
        print("=" * 80)

        batch_size = hparams["batch_size"]

        # Dataloaders for this trial
        train_loader = get_loader(train_X, train_y,
                                  batch_size=batch_size, shuffle=True)
        val_loader = get_loader(val_X, val_y,
                                batch_size=batch_size, shuffle=False)

        # Build model & optimizer
        config = MLPConfig(
            input_dim=input_dim,
            hidden_dims=hparams["hidden_dims"],
            output_dim=num_classes,
            dropout_rate=hparams["dropout"],
        )

        model = MLP(config).to(device)

        if hparams["use_class_weights"]:
            class_weights = compute_class_weights(train_y).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        # Train (uses early stopping inside)
        _ = train_loop(
            model=model,
            train_dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=args.max_epochs,
            val_dataloader=val_loader,
        )

        # Evaluate on validation set
        val_macro_f1 = evaluate_macro_f1(model, val_loader, device)
        print(f"Validation macro-F1: {val_macro_f1:.4f}")

        all_results.append(
            {
                "trial": trial,
                "hparams": hparams,
                "val_macro_f1": float(val_macro_f1),
            }
        )

        # Track best
        if val_macro_f1 > best_val_macro_f1:
            print(">>> New best model found!")
            best_val_macro_f1 = val_macro_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_hparams = hparams

        print()

    # Evaluate best model on test set
    print("\nHyperparameter search complete.")
    print("Best validation macro-F1: {:.4f}".format(best_val_macro_f1))
    print("Best hyperparameters:")
    for k, v in best_hparams.items():
        print(f"  {k}: {v}")

    # Rebuild best model and evaluate on test set
    best_model_config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=best_hparams["hidden_dims"],
        output_dim=num_classes,
        dropout_rate=best_hparams["dropout"],
    )
    best_model = MLP(best_model_config).to(device)
    best_model.load_state_dict(best_state_dict)

    best_batch_size = best_hparams["batch_size"]
    test_loader = get_loader(test_X, test_y,
                             batch_size=best_batch_size, shuffle=False)

    test_macro_f1 = evaluate_macro_f1(best_model, test_loader, device)
    print("\nTest macro-F1 of best model: {:.4f}".format(test_macro_f1))

    # Save artifacts
    results_path = "./results/hparam_search_results.json"
    torch.save(best_state_dict, "./results/best_model.pth")

    to_save = {
        "best_hyperparameters": best_hparams,
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_macro_f1": float(test_macro_f1),
        "all_trials": all_results,
        "label_mapping": {int(i): cls for i, cls in enumerate(le.classes_)},
    }

    with open(results_path, "w") as f:
        json.dump(to_save, f, indent=4)

    print(f"\nSaved best model to ./results/best_model.pth")
    print(f"Saved hyperparameter search results to {results_path}")


if __name__ == "__main__":
    main()
