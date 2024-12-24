import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import root_mean_squared_error
import random
import os


class MLP_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(np.array(X))  
        self.y = torch.Tensor(np.array(y)) 
        self.len=len(self.X)          

    def __getitem__(self, index):
        return self.X[index], self.y[index] 

    def __len__(self):
        return self.len

class MLP(torch.nn.Module):
    def __init__(self, seed, input_dim, num_layers, layer_depth, eta, gpu, folder):

        super().__init__()
        self.model = torch.nn.Sequential()
        self.model.append(torch.nn.Linear(input_dim,layer_depth))
        self.model.append(torch.nn.ReLU())
        for layer in range(num_layers-1):
            self.model.append(torch.nn.Linear(layer_depth,layer_depth))
            self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.Linear(layer_depth,1))

        self.eta = eta

        self.folder = folder
        self.num_train_epochs = 200 
        self.num_val_epochs = 100
        self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
    def fit(self, X_train, y_train, X_val, y_val):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.eta)        

        for epoch in range(self.num_train_epochs):
            epoch_loss = self.train(optimizer, X_train, y_train)

        best_val_loss = np.inf
        for epoch in range(self.num_val_epochs):
            train_loss = self.train(optimizer, X_train,y_train)
            val_loss = root_mean_squared_error(self.predict(X_val).cpu(),y_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(self.folder,"temp_best_MLP.pt"))
        
        checkpoint = torch.load(os.path.join(self.folder,"temp_best_MLP.pt"), weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        os.remove(os.path.join(self.folder,"temp_best_MLP.pt"))


    def train(self, optimizer, X_train, y_train):
        epoch_loss = []
        self.model.train() 

        train_data = MLP_Dataset(X_train, y_train)
        train_dataloader = DataLoader(train_data, batch_size=30, shuffle=True)

        for batch in train_dataloader:
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(X)

            loss = torch.nn.functional.mse_loss(y_pred,y.unsqueeze(1))
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.array(epoch_loss).mean()

    def predict(self, X):
        self.model.eval() 
        X_tensor = torch.Tensor(np.array(X)).to(self.device)
        return self.model(X_tensor).detach().flatten()

