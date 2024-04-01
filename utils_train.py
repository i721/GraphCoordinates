
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
from torch.utils.data import DataLoader, Subset, Dataset
from utils import saveLog
    
class MemoryMappedDataset(Dataset):
    def __init__(self, X_file, Y_file, transform=None, device=torch.device("cpu")):
        self.X = torch.load(X_file, map_location=device)
        self.Y = torch.load(Y_file, map_location=device)
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        saveLog(f"Created folder: {path}")

def copyFiles(source_folder, destination_folder):
    # Check if destination folder exists, if not create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Loop through all files in source folder and copy them to destination folder
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy2(source_file, destination_file)



class denseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=1, activation="relu",datasetName = "ogbn-products"):
        super(denseNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.gc_layers = nn.ModuleList()
        self.activation = activation
        self.datasetName = datasetName
        if len(self.hidden_dim) == 0:
            self.gc_layers.append(nn.Linear(input_dim, out_dim))
        else:
            self.gc_layers.append(nn.Linear(input_dim, hidden_dim[0]))
            for i in range(len(hidden_dim[1:])):
                if type(hidden_dim[i]) == int:
                    if type(hidden_dim[i+1]) == int:
                        self.gc_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                    else:
                        self.gc_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+2]))
                else:
                    self.gc_layers.append(torch.nn.Dropout(p=hidden_dim[i], inplace=False))
            self.gc_layers.append(nn.Linear(hidden_dim[-1], out_dim))
        # Cast the weights to a new dtype
        new_dtype = torch.float64
        for param in self.parameters():
            param.data = param.data.to(new_dtype)

    def forward(self, x):
        for i, layer in enumerate(self.gc_layers[:-1]):
            #x = torch.matmul(adj, x)
            x = layer(x)
            if type(self.hidden_dim[i]) == int:
                if self.activation == "relu":
                    x = F.relu(x)
                elif self.activation == "elu":
                    x = F.elu(x, alpha=1.0, inplace=False)
                #x = F.relu(x)
        x = self.gc_layers[-1](x)
        if self.datasetName == "ogbn-arxiv" or self.datasetName == "ogbn-products":
            x = F.log_softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x

class CustomNormalize(object):
    """Custom transform to apply normalization to tensor data"""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # normalize the tensor data
        tensor = (tensor - self.mean) / self.std
        return tensor

def to_one_hot(tensor, num_classes):
    # Get the shape of the input tensor
    n = tensor.shape[0]

    # Create a new tensor with the shape (n, num_classes)
    one_hot = torch.zeros(n, num_classes, dtype=torch.float32)

    # Fill the one_hot tensor with 1s at the indices specified by the input tensor
    tensor = tensor.to(torch.int64)
    one_hot.scatter_(1, tensor, 1)

    return one_hot
        
        
def evaluate_model(model, loader, evaluator, datasetName, datasetType='valid'):
    """
    Evaluate the model on a given dataset (validation or test).

    Parameters:
    - model: The neural network model to evaluate.
    - loader: DataLoader for the dataset to evaluate on.
    - evaluator: An Evaluator object for calculating the evaluation metric.
    - datasetName: Name of the dataset (e.g., 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins').
    - datasetType: Type of the dataset to evaluate ('valid' or 'test') for logging purposes.

    Returns:
    - result_dict: A dictionary containing the evaluation results.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            preds = model(X)
            all_preds.append(preds)
            all_labels.append(y)
    
    y_pred = torch.cat(all_preds, dim=0)
    y_test = torch.cat(all_labels, dim=0)

    # Adjust predictions for specific datasets
    if datasetName in ["ogbn-arxiv", "ogbn-products"]:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)
    
    input_dict = {"y_true": y_test, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)

    # Log the evaluation results
    if datasetType == 'valid':
        print(f"Validation Result: {result_dict}")
    elif datasetType == 'test':
        print(f"Test Result: {result_dict}")
    if datasetName == "ogbn-proteins":
        #print(f"ROCAUC: {result_dict['rocauc']}")
        return result_dict['rocauc']
    else:
        #print(f"Accuracy: {result_dict['acc']}")
        return result_dict['acc']

def train_epoch(model, train_loader, criterion, optimizer, device, datasetName):
    model.train()
    total_loss = 0
    total_samples = 0

    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = calculate_loss(y_pred.double(), y_train.double(), criterion, datasetName)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().cpu().numpy() * X_train.size(0)
        total_samples += X_train.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss

def calculate_loss(y_pred, y_train, criterion, datasetName):
    if datasetName in ["ogbn-arxiv", "ogbn-products"]:
        loss = criterion(y_pred, y_train)
    else:
        loss = criterion(y_pred, y_train)
    return loss

def main_train_loop(model, train_loader, valid_loader, optimizer, criterion, evaluator, datasetName, num_epochs, early_stop_patience, saveFolder, device=torch.device("cpu"), lr=0.01):
    best_valid_score = 0
    early_stop_counter = 0
    optimizer = optimizer(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, datasetName)
        valid_result = evaluate_model(model, valid_loader, evaluator, datasetName, datasetType='valid')
        
    
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Score = {valid_result:.4f}")
        
        # Early stopping logic
        if valid_result > best_valid_score:
            best_valid_score = valid_result
            early_stop_counter = 0
            print("New best score, saving model...")
            torch.save(model.state_dict(), f"{saveFolder}/best_model.pt")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"No improvement for {early_stop_patience} consecutive epochs, stopping early...")
                break

def load_data(X_file, Y_file, indices_file, batch_size, transform=None, device=torch.device("cpu")):
    dataset = MemoryMappedDataset(X_file, Y_file, transform=transform, device=device)
    indices = torch.load(indices_file)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

class CustomNormalize(object):
    """Custom transform to apply normalization to tensor data"""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # normalize the tensor data
        tensor = (tensor - self.mean) / self.std
        return tensor
    
class MemoryMappedDataset(Dataset):
    def __init__(self, X_file, Y_file, transform=None, device=torch.device("cpu")):
        self.X = torch.load(X_file, map_location=device)
        self.Y = torch.load(Y_file, map_location=device)
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

