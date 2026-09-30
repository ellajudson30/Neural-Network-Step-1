import numpy as np
import math
import random 
from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import csv
from rk45utils import run_RK45

# Create Pytorch Dataset 
class myDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x,y
class StandardizedDataset(myDataset):
    def __init__(self, data, labels, mean=None, std=None):
        self.labels = labels 

        # Calculate mean and std over the batch/rows dimension (dim=0) if not provided
        # ---> calculate on the training set and reuse for validation/testing)
        if mean is None or std is None:
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True)

            # Avoid division by zero for constant features
            self.std[self.std == 0] = 1.0 
        else:
            self.mean = mean
            self.std = std
            
        # Apply the standardization transform: (x - mean) / std
        self.data = (data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Possible network architectures
#-------------------------------------------------------------------------
class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16,32),
            nn.Tanh(), 
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

class NN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16,64),
            nn.Tanh(), 
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

class NN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16,64),
            nn.Tanh(), 
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

# Load Dataset
#-------------------------------------------------------------------------

# Open CSV file
with open('features5.csv', mode='r', newline='', encoding='utf-8') as file:
    # Create a reader object
    csv_reader = csv.reader(file)
    
    # Convert the reader object directly into a list
    Data_set = [[float(x) for x in row] for row in csv_reader]

with open('ratios5.csv', mode='r', newline='', encoding='utf-8') as file:
    # Create a reader object
    csv_reader = csv.reader(file)
    
    # Convert the reader object directly into a list
    Ratios = [[float(x) for x in row] for row in csv_reader]

print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )
print(f"Size of Target Ratios: {len(Ratios)}x{len(Ratios[0])}" )

# Diagnostics
# ------------------------------------------------------------------------
# Spread of ratios values in dataset

# ratios = np.array(Ratios).flatten()
# print("Number of targets:", len(ratios))
# print("min:", ratios.min())
# print("max:", ratios.max())

# print("r < 0.5:", np.sum(ratios < 0.5))
# print("0.5 <= r < 0.8:", np.sum((ratios >= 0.5) & (ratios < 0.8)))
# print("0.8 <= r < 1.2:", np.sum((ratios >= 0.8) & (ratios < 1.2)))
# print("1.2 <= r < 2:", np.sum((ratios >= 1.2) & (ratios < 2)))
# print("2 <= r < 5:", np.sum((ratios >= 2) & (ratios < 5)))
# print("r >= 5:", np.sum(ratios >= 5))

# normal_ratios = ratios[ratios < 2]
# plt.hist(normal_ratios, bins=50)
# plt.xlabel("PI Step Ratio")
# plt.ylabel("Number of Examples")
# plt.title("Distribution of Training Targets")
# plt.show()

# # Range of values for each feature
# X = np.array(Data_set)

# for i in range(X.shape[1]):
#     print(
#         f"Feature {i}: "
#         f"min = {X[:, i].min():.4f}, "
#         f"max = {X[:, i].max():.4f}, "
#         f"mean = {X[:, i].mean():.4f}, "
#         f"std = {X[:, i].std():.4f}"
#     )
# #----------------------------------------------------------------------------------------

# Convert Dataset into np.arrays
X = np.array(Data_set)
y = np.array(Ratios)

# Split into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# # Standardize
# train_mean = X_train.mean(axis=0, keepdims=True)
# train_std = X_train.std(axis=0, keepdims=True)

# train_std[train_std == 0] = 1.0

# X_train = (X_train - train_mean) / train_std
# X_test  = (X_test - train_mean) / train_std

# # Now convert to PyTorch datasets
# Training_data = myDataset(X_train, y_train)
# Testing_data = myDataset(X_test, y_test)

# # Split into training and test sets 
# Training_data, Testing_data = train_test_split(Dataset, test_size=0.2, random_state=42)

# Add standardizing here


# # Cross Validation - choose architecture
# #-------------------------------------------------------------------------------------
# kfold = KFold(n_splits=4, shuffle=True, random_state=42)

num_epochs = 100
# CV_MSE = []
# CV_MRE = []

# # CV on Training Data
# for fold, (train_index, val_index) in enumerate(kfold.split(X_train)):
    
#     # Split this fold
#     X_fold_train = X_train[train_index]
#     y_fold_train = y_train[train_index]

#     X_fold_val = X_train[val_index]
#     y_fold_val = y_train[val_index]

#     # Standardization
#     mean = X_fold_train.mean(axis=0, keepdims=True)
#     std = X_fold_train.std(axis=0, keepdims=True)

#     std[std == 0] = 1.0

#     X_fold_train = (X_fold_train - mean) / std
#     X_fold_val = (X_fold_val - mean) / std

#     # Make PyTorch datasets
#     TrainData = myDataset(X_fold_train, y_fold_train)
#     ValData = myDataset(X_fold_val, y_fold_val)

#     # TrainData = Subset(Training_data, train_index)
#     # ValData = Subset(Training_data, val_index)

#     batch = 16
#     Train_loader = DataLoader(TrainData, batch_size=batch, shuffle=True)
#     Val_loader = DataLoader(ValData, batch_size=batch)

#     # Define model, loss function and optimizer
#     # model = NN1()
#     # model = NN2()
#     model = NN3()
#     loss_fcn = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     train_losses = []
#     val_losses = []
#     train_loss = 0

#     # Training
#     for epoch in range(num_epochs):
#         model.train()

#         train_loss = 0
#         num_train = 0
#         for batch, (X,y) in enumerate(Train_loader):
#             y = y.float()
#             optimizer.zero_grad()
#             output = model(X)
#             loss = loss_fcn(output, y)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()*X.size(0)
#             num_train += X.size(0)
#         train_losses.append(train_loss/num_train)
    
#         # Validation
#         model.eval()

#         val_loss = 0.0
#         num_samples = 0
#         relative_err = 0

#         with torch.no_grad():
#             for batch, (X,y) in enumerate(Val_loader):
#                 y = y.float()
#                 test_output = model(X)
#                 loss = loss_fcn(test_output, y)

#                 # MSE
#                 val_loss += loss.item()*X.size(0)

#                 # Relative Error
#                 relative_err += torch.sum(torch.abs(test_output - y) / torch.abs(y)
#                                     ).item()

#                 num_samples += X.size(0)
#             val_losses.append(val_loss/num_samples)

#     #         predictions.append(test_output)
#     #         targets.append(y)

#     # for i in range(len(targets)):
#     #     err += abs(predictions[i]-targets[i])/targets[i]
#     # err = err/len(targets)
#     print(f"Fold {fold+1} MRE: ", relative_err/num_samples)
#     CV_MRE.append(relative_err/num_samples)

#     print(f"Fold {fold+1} MSE: ", val_loss/num_samples)
#     CV_MSE.append(val_loss/num_samples)

#     plt.plot(range(num_epochs),val_losses, label = "Val Loss")
#     plt.plot(range(num_epochs),train_losses, label = "Train Loss")
#     plt.legend()
#     plt.show()

# print("Average MSE: ", sum(CV_MSE)/len(CV_MSE), "+/-", np.std(CV_MSE))
# print("Average MRE: ", sum(CV_MRE)/len(CV_MRE), "+/-", np.std(CV_MRE))

# Choose final set-up and retrain on whole training data

# Standardize
train_mean = X_train.mean(axis=0, keepdims=True)
train_std = X_train.std(axis=0, keepdims=True)

train_std[train_std == 0] = 1.0

X_train = (X_train - train_mean) / train_std
X_test  = (X_test - train_mean) / train_std

# Now convert to PyTorch datasets
Training_data = myDataset(X_train, y_train)
Testing_data = myDataset(X_test, y_test)

batch = 32
Train_loader = DataLoader(Training_data, batch_size=batch, shuffle=True)
Test_loader = DataLoader(Testing_data, batch_size=batch)

model = NN3()
loss_fcn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_losses = []
targets = []
predictions = []
relative_err = []

# Training Loop
for epoch in range(num_epochs):
    model.train()

    train_loss = 0.0
    for batch, (X,y) in enumerate(Train_loader):
        y = y.float()
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fcn(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_losses.append(train_loss)

    # Testing
    model.eval()
    
    test_loss = 0.0
    num_test = 0
    err = 0.0
    targets = []
    predictions = []
    point_errors = []
    with torch.no_grad():
        for batch, (X,y) in enumerate(Test_loader):
            y = y.float()
            test_output = model(X)
            loss = loss_fcn(test_output, y)
            test_loss += loss.item() * X.size(0)
            num_test += X.size(0)

            # Compute Pointwise relative error to diagnose very poor preditions
            point_err = torch.abs(y - test_output) / torch.abs(y)
            point_errors.extend(point_err.flatten().cpu().numpy())

            targets.extend(y.flatten().cpu().numpy())
            predictions.extend(test_output.flatten().cpu().numpy())

        test_losses.append(test_loss/num_test)

        # Relative Error per epoch
        for i in range(len(targets)):
            err += abs(predictions[i]-targets[i])/targets[i]
        err = err/len(targets)
        relative_err.append(err.item())

point_errors = np.array(point_errors)
targets = np.array(targets)
predictions = np.array(predictions)

# print("Mean relative error:", np.mean(point_errors))
# print("Median relative error:", np.median(point_errors))
# print("90th percentile:", np.percentile(point_errors, 90))
# print("95th percentile:", np.percentile(point_errors, 95))
# print("Maximum relative error:", np.max(point_errors))
bins = [
    (0, 0.5),
    (0.5, 0.8),
    (0.8, 1.2),
    (1.2, 2.0),
    (2.0, 5.0),
    (5.0, np.inf)
]

for low, high in bins:

    mask = (targets >= low) & (targets < high)

    if np.sum(mask) > 0:

        mae = np.mean(
            np.abs(predictions[mask] - targets[mask])
        )

        mre = np.mean(
            np.abs(predictions[mask] - targets[mask])
            / np.abs(targets[mask])
        )

        print(
            f"{low} <= r < {high}: "
            f"n={np.sum(mask)}, "
            f"MAE={mae:.4f}, "
            f"MRE={mre:.4f}"
        )

plt.scatter(targets, point_errors, alpha=0.5)

low = min(targets.min(), predictions.min())
high = max(targets.max(), predictions.max())

plt.plot([low, high], [low, high], '--')

plt.xlabel("True PI Step Ratio")
plt.ylabel("NN Predicted Step Ratio")
plt.title("NN Predictions vs PI Controller")

plt.show()

plt.figure(1)
plt.plot(range(num_epochs),relative_err, label = "Relative Err")
# plt.plot(range(num_epochs),test_losses, label = "Test Loss")
# plt.legend()
plt.title('Relative Error')
plt.show()

plt.figure(2)
plt.plot(range(num_epochs),test_losses, label = "Test Loss")
# plt.legend()
plt.title('Test Loss (MSE)')
plt.show()

# # print(norm_err)
# # print(test_losses)

# # Save weights
# torch.save(model.state_dict(), 'model_weights_s1.pth')
# print("Model weights saved successfully!")
