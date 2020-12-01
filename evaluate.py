"""Script to load a model checkpoint and evaluate it on a new dataset.

Example
-------
python evaluate.py --load="saves_mlp4/window:10-inp:5-hid:20-bs:128-lr:0.001-std:0.005-seed:3/model_best.pt"
"""
import os
import ipdb
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import models
from dataset import MatRNNDataset, MatDNNDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Evaluate a trained model on a test set')
parser.add_argument('--load', type=str,
                    help='Path to the model checkpoint to load')
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


model = torch.load(args.load)
criterion = nn.MSELoss()

if isinstance(model, models.NeuralNet):
    # test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat", args.window_size)
    # test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_04.58 0degC_US06_Pan18650PF.mat", args.window_size)
    test_dataset = MatDNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_17.14 0degC_UDDS_Pan18650PF.mat", args.window_size)
else:
    # test_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat", args.sequence_length, args.window_size)
    # test_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_04.58 0degC_US06_Pan18650PF.mat", args.sequence_length, args.window_size)
    test_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_17.14 0degC_UDDS_Pan18650PF.mat", args.sequence_length, args.window_size)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
average_test_loss = evaluate(test_loader)
print('Test loss: {}'.format(average_test_loss))
