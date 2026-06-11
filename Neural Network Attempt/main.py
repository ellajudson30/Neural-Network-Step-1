import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Create training and test data
#-----------------------------------

# Custom Dataset
class FirstDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x,y

def fcn(x):
    return x + 0.1*math.exp(x)*math.cos(x)
def compute_labels(dataset, n):
    y = []
    for i in range(0,n):
        if fcn(dataset[i][0]) < dataset[i][1]:
            y.append(0)
        else:
            y.append(1)
    labels = torch.tensor(y, dtype=torch.long) 
    return labels 

n = 1024
intvl = [-np.pi, np.pi]
X_train = intvl[0]+(intvl[1]-intvl[0])*torch.rand(n, 2)
X_test = intvl[0]+(intvl[1]-intvl[0])*torch.rand(n, 2) 

train_labels = compute_labels(X_train, n)
test_labels = compute_labels(X_test, n)

train_data = FirstDataset(X_train, train_labels)
test_data = FirstDataset(X_test, test_labels)

batch = 16
train_dataloader = DataLoader(train_data, batch_size=batch)
test_dataloader = DataLoader(test_data, batch_size=batch)

# Define Model
#----------------------------------
class FirstNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,16),
            nn.Tanh(),
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output
    
# Loss function and Optimizer
model = FirstNN()
loss_fcn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Functions for Training 
#-----------------------------------
def compute_accuracy(output, labels):
    labels = labels.view(-1, 1)
    pred = (output >= 0).float()
    true_pos = ((pred == 1) & (labels == 1)).sum().item()
    true_neg = ((pred == 0) & (labels == 0)).sum().item()
    acc = (true_pos+true_neg)/labels.numel()

    return acc

def train_one_epoch(train_dataloader):
    running_loss=0.0
    running_acc=0.0

    for batch, (X,y) in enumerate(train_dataloader):
        y = y.float().unsqueeze(1)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fcn(output, y)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(output,y)
        running_acc += acc
        running_loss += loss.item()
    
    avg_acc = running_acc/len(train_dataloader)

    return running_loss, avg_acc

def test_one_epoch(test_dataloader):
    test_running_loss=0.0
    test_running_acc=0.0

    with torch.no_grad():
        for batch, (X,y) in enumerate(test_dataloader):
            y = y.float().unsqueeze(1)
            test_output = model(X)
            loss = loss_fcn(test_output, y)

            acc = compute_accuracy(test_output,y)
            test_running_acc += acc
            test_running_loss += loss.item()
    
    test_avg_acc = test_running_acc/len(test_dataloader)

    return test_running_loss, test_avg_acc


#Training Loop
#-----------------------------------
num_epochs = 20
train_losses = []
test_losses = []

train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    # Train
    model.train()
    loss, acc = train_one_epoch(train_dataloader)
    train_losses.append(loss)
    train_accuracy.append(acc)
    
    # Test
    model.eval()
    test_loss, test_acc = test_one_epoch(test_dataloader)
    test_losses.append(test_loss)
    test_accuracy.append(test_acc)

# print(train_losses)
# print(test_losses)
# print(train_accuracy)
# print(test_accuracy)

# Plot the Losses
plt.figure(1)
plt.plot(range(num_epochs), train_losses, label = "Train Loss")
plt.plot(range(num_epochs), test_losses, label = "Test Loss")
plt.title("Training and Test Losses")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot the Accuracy
plt.figure(2)
plt.plot(range(num_epochs), train_accuracy, label="Train Accuracy")
plt.plot(range(num_epochs), test_accuracy, label="Test Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Visualize the decision boundary of the NN
#---------------------------------------------

# Exact Decision Boundary
grid_size = 100
x = np.linspace(-np.pi, np.pi, grid_size)
f_exact = x + 0.1*np.exp(x)*np.cos(x)

# Create grid
vec = torch.linspace(-np.pi, np.pi, grid_size)
XX, YY = torch.meshgrid(vec, vec, indexing='ij')
grid_pts = torch.stack([XX,YY], dim = -1).view(-1,2)

# Compute labels for grid points (to compute accuracy)
grid_labels = compute_labels(grid_pts, grid_size**2)

# Pass through model
model.eval()
with torch.no_grad():
    predictions = model(grid_pts) 
predict_grid = predictions.reshape(100, 100)

# Plot Curves
plt.figure(3)
plt.contour(XX,YY,predict_grid, levels=[0.5], colors='red', linewidths=2)
contour_handle = Line2D([], [], color='red', linewidth=2)
exact_line, = plt.plot(x,f_exact, color='blue', linewidth=2)

plt.title("Decision Boundary: Exact vs Learned")
plt.legend([contour_handle, exact_line],
    ["NN Boundary", "Exact Boundary"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

