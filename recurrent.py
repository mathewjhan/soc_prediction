"""Train an LSTM on the SOC data.

Example
-------
python lstm.py --sequence_length 100 --hidden_size 4 --num_layers 4 --noise_std 0.005
"""
import os
import sys
import ipdb
import random
import argparse
import pickle as pkl

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('bright')

import models
from dataset import MatRNNDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
parser = argparse.ArgumentParser(description='LSTM')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='Number of epochs to train')
parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'rnn'],
                    help='Choose the type of model')
parser.add_argument('--sequence_length', type=int, default=100,
                    help='Sequence length')
parser.add_argument('--window_size', type=int, default=0,
                    help='Window size')
parser.add_argument('--input_size', type=int, default=3,
                    help='Input size')
parser.add_argument('--hidden_size', type=int, default=10,
                    help='Hidden size')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of layers')
parser.add_argument('--num_classes', type=int, default=1,
                    help='Number of classes')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--noise_std', type=float, default=0.0,
                    help='Noise standard deviation')
parser.add_argument('--save_dir', type=str, default='rnn_saves',
                    help='Base save directory')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
args = parser.parse_args()

# Set random seed for reproducibility
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

exp_name = 'm:{}-seq:{}-window:{}-inp:{}-hid:{}-lay:{}-bs:{}-lr:{}-std:{}-seed:{}'.format(
            args.model, args.sequence_length, args.window_size, args.input_size,
            args.hidden_size, args.num_layers, args.batch_size,
            args.learning_rate, args.noise_std, args.seed)

save_dir = os.path.join(args.save_dir, exp_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dataset
dataset_dir = "data/Panasonic 18650PF Data/0degC/Drive cycles/"
train_files = [
               "05-30-17_12.56 0degC_Cycle_1_Pan18650PF.mat",
               "05-30-17_20.16 0degC_Cycle_2_Pan18650PF.mat",
               "06-01-17_15.36 0degC_Cycle_3_Pan18650PF.mat",
               "06-01-17_22.03 0degC_Cycle_4_Pan18650PF.mat"
              ]
train_datasets = list()
for f in train_files:
    temp = MatRNNDataset(dataset_dir + f, args.sequence_length, args.window_size)
    train_datasets.append(temp)

# train_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-01-17_22.03 0degC_Cycle_4_Pan18650PF.mat", args.sequence_length, args.window_size)
validation_dataset = MatRNNDataset("data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat", args.sequence_length, args.window_size)

train_loaders = list()
for d in train_datasets:
    temp = DataLoader(dataset=d, batch_size=args.batch_size, shuffle=True)
    train_loaders.append(temp)

# train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size, shuffle=False)

# Model, loss, and optimizer
if args.model == 'lstm':
    model = models.LSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes, args.noise_std).to(device)
elif args.model == 'gru':
    model = models.GRU(args.input_size, args.hidden_size, args.num_layers, args.num_classes, args.noise_std).to(device)
elif args.model == 'rnn':
    model = models.RNN(args.input_size, args.hidden_size, args.num_layers, args.num_classes, args.noise_std).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train the model
# total_step = len(train_loader)
# total_train_step = len(train_loader)
# total_val_step = len(validation_loader)

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

epochs = list()
train_losses = list()
val_losses = list()

best_val_loss = 1e9  # An initial large value so that we always do better than this

for epoch in range(args.num_epochs):

    total_train_loss = 0
    total_val_loss = 0
    num_train_batches = 0
    num_val_batches = 0

    model.train()
    for loader in train_loaders:
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update train_loss, items
            total_train_loss += loss.item()
            num_train_batches += 1

            # if (i+1) % 100 == 0:
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #            .format(epoch+1, args.num_epochs, i+1, total_train_step, loss.item()))

    average_train_loss = total_train_loss / num_train_batches
    average_val_loss = evaluate(validation_loader)

    epochs.append(epoch)
    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model, os.path.join(save_dir, 'model_best.pt'))

    torch.save(model, os.path.join(save_dir, 'model_latest.pt'))

    with open(os.path.join(save_dir, 'result.pkl'), 'wb') as f:
        pkl.dump({'epochs': epochs,
                  'train_losses': train_losses,
                  'val_losses': val_losses }, f)

    print("Epoch: {} | Train loss: {} | Val loss: {}".format(epoch, average_train_loss, average_val_loss))
    sys.stdout.flush()

# Test the model
# model.eval()
# with torch.no_grad():
#     total = 0
#     error = 0
#     for x, y in validation_loader:
#         x = x.to(device)
#         y = y.to(device)
#
#         outputs = model(x)
#         error_tensor = torch.sum(torch.abs(y - outputs.data))
#         error += error_tensor.item()
#         total += y.size(0)
#
#     print("Validation average percentage error in " + str(total) + " samples: " + "{:.2f}".format(error/total*100) + "%")


with open(os.path.join(save_dir, 'result.pkl'), 'wb') as f:
    pkl.dump({'epochs': epochs,
              'train_losses': train_losses,
              'val_losses': val_losses }, f)


print("Final train loss:", train_losses[-1])
print("Final validation loss:", val_losses[-1])

plt.figure()
plt.plot(epochs, train_losses, linewidth=2, label='Train')
plt.plot(epochs, val_losses, linewidth=2, label='Val')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=18)
plt.yscale('log')
plt.savefig(os.path.join(save_dir, 'LSTM.pdf'), bbox_inches='tight', pad_inches=0)

# Save the model checkpoint
# torch.save(model.state_dict(), os.path.join(save_dir, 'rnn.ckpt'))
torch.save(model, os.path.join(save_dir, 'rnn.pt'))
