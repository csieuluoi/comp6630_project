#%%
import os
# os.chdir('/data/dungp/projects/COMP6630/project')
import argparse

parser = argparse.ArgumentParser(description='Train MLP model for cell type classification')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


from model import MLP, MLPConfig, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import anndata as ad
import pandas as pd

def load_data():

    train_X = np.load('/data/dungp/projects/COMP6630/project/data/processed_data/train_X.npy')
    train_y = np.load('/data/dungp/projects/COMP6630/project/data/processed_data/train_y.npy', allow_pickle=True)
    val_X = np.load('/data/dungp/projects/COMP6630/project/data/processed_data/val_X.npy')
    val_y = np.load('/data/dungp/projects/COMP6630/project/data/processed_data/val_y.npy', allow_pickle=True)
    test_X = np.load('/data/dungp/projects/COMP6630/project/data/processed_data/test_X.npy')
    test_y = np.load('/data/dungp/projects/COMP6630/project/data/processed_data/test_y.npy', allow_pickle=True)

    return train_X, train_y, val_X, val_y, test_X, test_y

def get_loader(X, y, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_loop(model, train_dataloader, criterion, optimizer, device, n_epochs, val_dataloader = None):
    earlystop = EarlyStopping(patience=5, min_delta=0.001)
    model.train()
    losses = {"train": [], "val": []}
    for epoch in range(n_epochs):
        total_loss = 0

        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        losses['train'].append(avg_loss)
        
        if val_dataloader is not None:
            # Validate the model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{n_epochs}], Val Loss: {avg_val_loss:.4f}")
            losses['val'].append(avg_val_loss)
            earlystop(avg_val_loss)
            if earlystop.early_stop:
                print("Early stopping")
                break
            
    print("Training complete.")
    return losses

def evaluate(model, dataloader, device, le = None):
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            labels.extend(label.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    if le is not None:
        labels = le.inverse_transform(labels)
        predictions = le.inverse_transform(predictions)
    print(classification_report(labels, predictions))
    return classification_report(labels, predictions, output_dict=True)

def extract_features(model, dataloader, device):
    model.eval()
    features = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            output = model.extract_features(inputs)
            features.append(output.cpu().numpy())

    features = np.vstack(features)
    return features
#%%
def main():
    # Hyperparameters
    dropout_rate = 0.5
    learning_rate = 0.001
    batch_size = 512
    n_epochs = 20
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_X, train_y, val_X, val_y, test_X, test_y = load_data()
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)
    val_y = le.transform(val_y)
    test_y = le.transform(test_y)

    
    train_loader = get_loader(train_X, train_y, batch_size=batch_size, shuffle=True)
    val_loader = get_loader(val_X, val_y, batch_size=batch_size, shuffle=False)

    input_dim  = train_X.shape[1]
    output_dim = len(le.classes_)  
    # Initialize model, loss function, and optimizer
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=output_dim,
    )

    model = MLP(config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_loop(model, train_loader, criterion, optimizer, device, n_epochs, val_dataloader=val_loader)

    # Evaluate the model
    test_loader = get_loader(test_X, test_y, batch_size=batch_size, shuffle=False)

    test_embeddings = extract_features(model, test_loader, device)

    ## create h5ad file for the test embeddings
    
    test_adata = ad.AnnData(X=test_embeddings)
    test_adata.obs['label'] = pd.Categorical.from_codes(test_y, le.classes_)
    test_adata.obs['predicted_label'] = pd.Categorical.from_codes(np.argmax(model(torch.tensor(test_X, dtype=torch.float32).to(device)).detach().cpu().numpy(), axis=1), le.classes_)
    
    test_adata.write_h5ad('./results/test_embeddings.h5ad')
    test_clf_report = evaluate(model, test_loader, device, le=le)
    ## save the classification report
    with open('./results/classification_report.json', 'w') as f:
        json.dump(test_clf_report, f, indent=4)

if __name__ == "__main__":
    main()