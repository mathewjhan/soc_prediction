"""Script to load all model checkpoints in a directory, and evaluate them on a new dataset.

Example
-------
python evaluate_seeds.py --load="saves_mlp4/window:10-inp:5-hid:20-bs:128-lr:0.001-std:0.005-seed:3/model_best.pt"
"""
import os
import ipdb
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import models
from dataset import MatRNNDataset, MatDNNDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Evaluate a trained model on a test set')
parser.add_argument('--load_dir', type=str,
                    help='Path to the directory of model checkpoints to load')
parser.add_argument('--sequence_length', type=int, default=100,
                    help='Sequence length')
parser.add_argument('--window_size', type=int, default=10,
                    help='Window size')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
args = parser.parse_args()

def evaluate(dataloader):
    total_loss = 0.0
    num_batches = 0.0
    model.eval()
    with torch.no_grad():
        for (x,y) in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches

criterion = nn.MSELoss()
all_test_losses = []

exp_dirs = os.listdir(args.load_dir)
for exp_dir in exp_dirs:
    model = torch.load(os.path.join(args.load_dir, exp_dir, 'model_best.pt'))

    if isinstance(model, models.NeuralNet):
        # test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat", args.window_size)
        # test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_04.58 0degC_US06_Pan18650PF.mat", args.window_size)
        test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_17.14 0degC_UDDS_Pan18650PF.mat", args.window_size)
    else:
        # test_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat", args.sequence_length, args.window_size)
        # test_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_04.58 0degC_US06_Pan18650PF.mat", args.sequence_length, args.window_size)
        test_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_17.14 0degC_UDDS_Pan18650PF.mat", args.sequence_length, args.window_size)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loss = evaluate(test_loader)
    all_test_losses.append(test_loss)
    print('Test loss: {}'.format(test_loss))

print('Mean test loss: {:6.3e} | Std test loss: {:6.3e}'.format(np.mean(all_test_losses), np.std(all_test_losses)))
