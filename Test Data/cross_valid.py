import numpy as np
import math
import random 
from scipy.integrate import RK45
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

class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11,32),
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
            nn.Linear(11,24),
            nn.Tanh(), 
            nn.Linear(24,24),
            nn.Tanh(),
            nn.Linear(24,1)
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

# Open the CSV file
with open('features.csv', mode='r', newline='', encoding='utf-8') as file:
    # Create a reader object
    csv_reader = csv.reader(file)
    
    # Convert the reader object directly into a list
    Data_set = [[float(x) for x in row] for row in csv_reader]

with open('ratios.csv', mode='r', newline='', encoding='utf-8') as file:
    # Create a reader object
    csv_reader = csv.reader(file)
    
    # Convert the reader object directly into a list
    Ratios = [[float(x) for x in row] for row in csv_reader]

print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )
print(f"Size of Target Ratios: {len(Ratios)}x{len(Ratios[0])}" )

Dataset = myDataset(Data_set, Ratios)

# Split into training and test sets 
Training_data, Testing_data = train_test_split(Dataset, test_size=0.2, random_state=42)

# print(len(Training_data))
# print(len(Testing_data))

kfold = KFold(n_splits=4, shuffle=True, random_state=42)

num_epochs = 10
CV_results = []

# Do CV on Training Data
for fold, (train_index, val_index) in enumerate(kfold.split(Training_data)):
    
    TrainData = Subset(Dataset, train_index)
    ValData = Subset(Dataset, val_index)

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


# plt.plot(range(num_epochs),norm_err, label = "abs err")
# plt.plot(range(num_epochs),test_losses, label = "Test Loss")
# plt.legend()
# plt.show()

# print(norm_err)
# print(test_losses)

# Test trained NN 

def testode1(t,y):
    return 4*y*(1-y)
y0 = [0.5]
t_span = [0, 5]
tol = 1e-10

def testode2(t,y):
    return y*(1-y)

def run_RK45_NN(fcn, t0, y0, tf, tol, model, numhist):

    solver = RK45(fcn,t0, y0, tf, rtol=tol, atol=tol)
    
    # Storage arrays
    times = []
    solution = []
    time_steps = []
    errors = []
    nn_steps = []
    # stage_history = []

    while solver.t < tf:

        solution.append(solver.y[0]) # only works if one IC passed
        times.append(float(solver.t))
        
        if len(time_steps) > numhist+1: 

            feature = []

            for i in range(numhist,-1,-1):
                feature.append(float(solution[-1-i]))
            
            for j in range(numhist, -1,-1):
                ratio = time_steps[-1-j]/time_steps[-2-j]
                feature.append(ratio)
            feature.append(float(np.log(tol)))

            feat_ten = torch.tensor(feature, dtype=torch.float32)

            with torch.no_grad():
                model.eval()
                next_h = model(feat_ten.unsqueeze(0)).item()
            
            solver.h_abs = solver.h_abs * next_h
            # nn_steps.append(solver.h_abs)
        
        # prevent overshoot of tf - does not work as needed
        remaining = tf - solver.t
        if solver.h_abs > remaining:
            solver.h_abs = remaining

        nn_steps.append(solver.h_abs) # to check if RK45 adjusted the step
        t_old = solver.t
        
        solver.step()
        h = solver.t - t_old
        time_steps.append(float(h))
        # stage_history.append(solver.K.copy())

        scale = solver.atol + solver.rtol * np.maximum(np.abs(solver.y_old), 
                                                    np.abs(solver.y))
        
        err = solver._estimate_error_norm(solver.K, h, scale)

        errors.append(float(err))

    return times, solution, time_steps, errors, nn_steps

times, sol, ts, errors, sh = run_RK45(testode1,t_span[0], y0, t_span[1], tol)
times_nn, sol_nn, ts_nn, errors_nn, nn_steps = run_RK45_NN(testode1,t_span[0], y0, t_span[1], tol, model, 4)
# print(ts)
# print(ts_nn)
print(len(ts))
print(len(ts_nn))

predicted = np.array(ts_nn[5:100])
actual = np.array(ts[5:100])
rmse = np.sqrt(np.mean((predicted-actual)**2))
rmse_steps = np.sqrt(np.mean((predicted-np.array(nn_steps)[5:100])**2))
print(rmse)
print(rmse_steps)

# plt.plot(ts[5:])
# plt.plot(ts_nn[5:])
# plt.show()

 