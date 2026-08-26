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

# Possible network architectures
#-------------------------------------------------------------------------
class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11,48),
            nn.Tanh(), 
            nn.Linear(48,48),
            nn.Tanh(),
            nn.Linear(48,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

class NN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11,64),
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
            nn.Linear(11,16),
            nn.Tanh(), 
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

# Load Dataset
#-------------------------------------------------------------------------

# Open CSV file
with open('features2.csv', mode='r', newline='', encoding='utf-8') as file:
    # Create a reader object
    csv_reader = csv.reader(file)
    
    # Convert the reader object directly into a list
    Data_set = [[float(x) for x in row] for row in csv_reader]

with open('ratios2.csv', mode='r', newline='', encoding='utf-8') as file:
    # Create a reader object
    csv_reader = csv.reader(file)
    
    # Convert the reader object directly into a list
    Ratios = [[float(x) for x in row] for row in csv_reader]

print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )
print(f"Size of Target Ratios: {len(Ratios)}x{len(Ratios[0])}" )

# Convert into Pytorch Dataset
Dataset = myDataset(Data_set, Ratios)

# Split into training and test sets 
Training_data, Testing_data = train_test_split(Dataset, test_size=0.2, random_state=42)

# Cross Validation - choose architecture
#-------------------------------------------------------------------------------------
kfold = KFold(n_splits=4, shuffle=True, random_state=42)

num_epochs = 10
CV_results = []

# CV on Training Data
for fold, (train_index, val_index) in enumerate(kfold.split(Training_data)):
    
    TrainData = Subset(Training_data, train_index)
    ValData = Subset(Training_data, val_index)

    batch = 16
    Train_loader = DataLoader(TrainData, batch_size=batch)
    Val_loader = DataLoader(ValData, batch_size=batch)

    # Define model, loss function and optimizer
    model = NN1()
    # model = NN2()
    # model = NN3()
    loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(num_epochs):
        model.train()

        for batch, (X,y) in enumerate(Train_loader):
            y = y.float()
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fcn(output, y)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch, (X,y) in enumerate(Val_loader):
            y = y.float()
            test_output = model(X)
            loss = loss_fcn(test_output, y)

            val_loss += loss.item()

    print(f"Fold {fold+1} MSE: ", val_loss)
    CV_results.append(val_loss)

print("Average MSE: ", sum(CV_results)/4)

# Choose final set-up and retrain on whole training data
batch = 32
Train_loader = DataLoader(Training_data, batch_size=batch)
Test_loader = DataLoader(Testing_data, batch_size=batch)

model = NN1()
loss_fcn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_losses = []
targets = []
predictions = []
norm_err = []

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
    err = 0.0
    targets = []
    predictions = []
    with torch.no_grad():
        for batch, (X,y) in enumerate(Test_loader):
            y = y.float()
            test_output = model(X)
            loss = loss_fcn(test_output, y)
            test_loss += loss.item()
            
            targets.extend(y)
            predictions.extend(test_output)
        test_losses.append(test_loss)
        
        for i in range(len(targets)):
            err += abs(predictions[i]-targets[i])/targets[i]
        err = err/len(targets)
        norm_err.append(err.item())

torch.save(model.state_dict(), 'model_weights_s1.pth')
print("Model weights saved successfully!")
# plt.plot(range(num_epochs),norm_err, label = "abs err")
# plt.plot(range(num_epochs),test_losses, label = "Test Loss")
# plt.legend()
# plt.show()

# print(norm_err)
# print(test_losses)
