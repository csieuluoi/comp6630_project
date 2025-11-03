import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: list  # List of hidden layer dimensions
    output_dim: int
    dropout_rate: float = 0.5

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        layers = []
        input_dim = config.input_dim

        # Create hidden layers 
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim

        # Add the output layer
        layers.append(nn.Linear(input_dim, config.output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    ## extract features from the network
    def extract_features(self, x):
        for layer in self.model[:-1]: 
            x = layer(x)
        return x


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0