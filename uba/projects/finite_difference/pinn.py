import time
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pyDOE import lhs

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def getBottomBC(n: int=50, value: float=0):
    x = lhs(1, n)
    bc = np.concatenate((x, np.zeros((n, 1)), value * np.ones((n, 1))), axis=1)
    return bc[:, 0:2], bc[:, 2:]

def getTopBC(n: int=50, value: float=0):
    x = lhs(1, n)
    bc = np.concatenate((x, np.ones((n, 1)), value * np.ones((n, 1))), axis=1)
    return bc[:, 0:2], bc[:, 2:]

def getLeftBC(n: int=50, value: float=0):
    y = lhs(1, n)
    bc = np.concatenate((np.zeros((n, 1)), y, value * np.ones((n, 1))), axis=1)
    return bc[:, 0:2], bc[:, 2:]

def getRightBC(n: int=50, value: float=0):
    y = lhs(1, n)
    bc = np.concatenate((np.ones((n, 1)), y, value * np.ones((n, 1))), axis=1)
    return bc[:, 0:2], bc[:, 2:]

def getBC(n: int=50, values: list=[0, 0, 0, 0], active: list=[True, True, True, True]):
    # Both Dirichlet and Neumann BCs can be built using this function (homogeneous by default)
    # Order: bottom, top, left, right
    x, y = lhs(1, n), lhs(1, n)

    # Boundary conditions
    bc_bottom = np.concatenate((x, np.zeros((n, 1)), values[0] * np.ones((n, 1))), axis=1)
    bc_top = np.concatenate((x, np.ones((n, 1)), values[1] * np.ones((n, 1))), axis=1)
    bc_left = np.concatenate((np.zeros((n, 1)), y, values[2] * np.ones((n, 1))), axis=1)
    bc_right = np.concatenate((np.ones((n, 1)), y, values[3] * np.ones((n, 1))), axis=1)

    bc = []
    bc += [bc_bottom] if active[0] else []
    bc += [bc_top] if active[1] else []
    bc += [bc_left] if active[2] else []
    bc += [bc_right] if active[3] else []

    bc = np.concatenate(bc, axis=0)
    return bc[:, 0:2], bc[:, 2:]

def getCollocationPoints(n: int=50):
    return lhs(2, n)

class Supervised(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
class SupervisedDM(pl.LightningDataModule):
    def __init__(self, xy: np.ndarray, u: np.ndarray, batch_size: int=32):
        super().__init__()
        self.n = xy.shape[0]
        self.xy = torch.tensor(xy, dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32)
        self.batch_size = batch_size
    
    def setup(self, stage: str=None, percent: float=0.9):
        indices = np.random.permutation(self.n)
        idx_lim = int(percent * self.n)
        # Get the indices for the train, validation and test sets
        train_idx = indices[0:idx_lim]
        val_idx = indices[idx_lim:]
        # Split the data into train, validation and test
        self.xy_train = self.xy[train_idx]
        self.xy_val = self.xy[val_idx]
        # Split the labels into train, validation and test
        self.u_train = self.u[train_idx]
        self.u_val = self.u[val_idx]
    
    def getTrainDataloader(self):
        train_split = Supervised(self.xy_train, self.u_train)
        return DataLoader(train_split, batch_size=self.batch_size, shuffle=True)

    def getValDataloader(self):
        val_split = Supervised(self.xy_val, self.u_val)
        return DataLoader(val_split, batch_size=self.batch_size)
    
class Unsupervised(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx]
    
class UnsupervisedDM(pl.LightningDataModule):
    def __init__(self, xy: np.ndarray, batch_size: int=32):
        super().__init__()
        self.n = xy.shape[0]
        self.xy = torch.tensor(xy, dtype=torch.float32, requires_grad=True)
        self.batch_size = batch_size
    
    def setup(self, stage: str=None, percent: float=0.9):
        indices = np.random.permutation(self.n)
        idx_lim = int(percent * self.n)
        # Get the indices for the train, validation and test sets
        train_idx = indices[0:idx_lim]
        val_idx = indices[idx_lim:]
        # Split the data into train, validation and test
        self.xy_train = self.xy[train_idx]
        self.xy_val = self.xy[val_idx]
    
    def getTrainDataloader(self):
        train_split = Unsupervised(self.xy_train)
        return DataLoader(train_split, batch_size=self.batch_size, shuffle=True)

    def getValDataloader(self):
        val_split = Unsupervised(self.xy_val)
        return DataLoader(val_split, batch_size=self.batch_size)

def getSupervisedDataLoader(getData, n: int=100, batch_size: int=32, **kwargs):
    if getData != getCollocationPoints:
        values = kwargs.get('values', [0, 0, 0, 0])
        active = kwargs.get('active', [True, True, True, True])
        xy, u = getData(n, values=values, active=active)
    else:
        xy, u = getData(n)
    dataset = SupervisedDM(xy, u, batch_size=batch_size)
    dataset.setup()
    train_dl = dataset.getTrainDataloader()
    val_dl = dataset.getValDataloader()
    return (train_dl, val_dl)

def getUnsupervisedDataLoader(getData, n: int=100, batch_size: int=32, **kwargs):
    if getData != getCollocationPoints:
        values = kwargs.get('values', [0, 0, 0, 0])
        active = kwargs.get('active', [True, True, True, True])
        xy = getData(n, values=values, active=active)
    else:
        xy = getData(n)
    dataset = UnsupervisedDM(xy, batch_size=batch_size)
    dataset.setup()
    train_dl = dataset.getTrainDataloader()
    val_dl = dataset.getValDataloader()
    return (train_dl, val_dl)

class PoissonPINN(nn.Module):
    def __init__(
            self, 
            sizes: list, 
            activations: list,
            loss_fn = nn.MSELoss(),  
            init_type: str='xavier', 
            device: str='cpu',
    ):
        super(PoissonPINN, self).__init__()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.loss_fn = loss_fn
        self.device = device

        # Define the layers.
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(activations):
                layers.append(activations[i])
        self.layers = nn.Sequential(*layers)

        # Initialize the parameters.
        if init_type is not None:
            self.initialize_weights(init_type)

        # Define the metrics.
        self.metrics = {
            'epochs': [], 
            'loss': {
                'train': {'bc': [], 'domain': [], 'data': [], 'total': []}, 
                'eval' : {'bc': [], 'domain': [], 'data': [], 'total': []}
            }, 
            'time': 0.0
        }

    def forward(self, xy):
        return self.layers(xy)
    
    def initialize_weights(self, init_type):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'he':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, a=0, b=1)

    def computeSupervisedLoss(self, input_batch, target_batch, gradient: bool=False, axis: int=0):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        u_pred_batch = self.forward(input_batch)
        if gradient:
            du_dxy = torch.autograd.grad(u_pred_batch, input_batch, torch.ones_like(u_pred_batch))[0]
            du_dn = du_dxy[:, axis:axis+1]
            loss = self.loss_fn(du_dn, target_batch)
        else:
            loss = self.loss_fn(u_pred_batch, target_batch)
        return loss
    
    def computeResidualPDELoss(self, xy_batch):
        xy_batch = xy_batch.to(self.device)
        x, y = xy_batch[:, 0:1], xy_batch[:, 1:2]
        u = self.forward(xy_batch)
        u_xy = torch.autograd.grad(u, xy_batch, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]
        u_xx = torch.autograd.grad(u_x, xy_batch, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy_batch, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        laplacian = u_xx + u_yy                         # compute laplacian
        source = self.computeSource(x, y, u)            # compute source term
        f = laplacian - source                          # compute residual
        return self.loss_fn(f, torch.zeros_like(f))     # compute loss
    
    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str="model_params.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path: str="model_params.pth"):
        self.load_state_dict(torch.load(path))
        self.to(self.device)

    def plotLoss(self, figsize=(10, 5), save: bool=False, filename: str="loss.png"):
        plt.figure(figsize=figsize)
        plt.plot(self.metrics['epochs'], self.metrics['train_loss'], label='Training')
        if self.metrics['eval_loss']:
            plt.plot(self.metrics['epochs'], self.metrics['eval_loss'], label='Validation')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        if save:
            plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w')
        plt.show()

class NonLinearPoissonPINN(PoissonPINN):
    
    def __init__(
            self, 
            sizes: list, 
            activations: list,
            loss_fn = nn.MSELoss(),
            init_type: str='xavier',
            device: str='cpu',
    ):
        super(NonLinearPoissonPINN, self).__init__(sizes, activations, loss_fn, init_type, device)

    def trainBatch(self, bc_dirichlet_batch, bc_right_batch, bc_top_batch, domain_batch, optimizer):
        xy_dirichlet_batch, u_dirichlet_batch = bc_dirichlet_batch
        xy_right_batch, du_right_batch = bc_right_batch
        xy_top_batch, du_top_batch = bc_top_batch
        loss_bc  = self.computeSupervisedLoss(xy_dirichlet_batch, u_dirichlet_batch)
        loss_bc += self.computeSupervisedLoss(xy_right_batch, du_right_batch, gradient=True, axis=0)
        loss_bc += self.computeSupervisedLoss(xy_top_batch, du_top_batch, gradient=True, axis=1)
        loss_domain = self.computeResidualPDELoss(domain_batch)
        loss = loss_bc + loss_domain
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss_bc.item(), loss_domain.item(), loss.item()

    def computeSource(self, x, y, u):
        return 0.5 * torch.exp(u)
    
    def computeLoss(self, bc_dirichlet_dl, bc_right_dl, bc_top_dl, domain_dl):
        # Compute dirichlet bc loss
        bc_loss = 0.0
        ## Dirichlet BC
        for xy_batch, u_batch in bc_dirichlet_dl:
            bc_loss += self.computeSupervisedLoss(xy_batch, u_batch).item()
        ## Neumann BC
        for xy_batch, du_batch in bc_right_dl:
            bc_loss += self.computeSupervisedLoss(xy_batch, du_batch, gradient=True, axis=0).item()
        for xy_batch, du_batch in bc_top_dl:
            bc_loss += self.computeSupervisedLoss(xy_batch, du_batch, gradient=True, axis=1).item()
        
        # Compute domain loss
        domain_loss = 0.0
        for xy_batch in domain_dl:
            domain_loss += self.computeResidualPDELoss(xy_batch).item()

        # Compute total loss
        total_loss = bc_loss + domain_loss

        # Normalize the losses
        bc_loss /= len(bc_dirichlet_dl) + len(bc_right_dl) + len(bc_top_dl)
        domain_loss /= len(domain_dl)
        total_loss /= len(bc_dirichlet_dl) + len(bc_right_dl) + len(bc_top_dl) + len(domain_dl)
        
        return bc_loss, domain_loss, total_loss
    
    def fit(self, train_dataloader, optimizer=optim.Adam, epochs=30, lr=1e-4, 
        regularization=0.0, eval_dataloader=None, verbose=True, epch_print=1):

        # Set the starting epoch
        last_epoch = self.metrics['epochs'][-1] if self.metrics['epochs'] else 0
        starting_epoch = last_epoch + 1
    
        # Get the dataloaders
        dirichlet_train_dl, right_train_dl, top_train_dl, domain_train_dl = train_dataloader
        dirichlet_eval_dl, right_eval_dl, top_eval_dl, domain_eval_dl = eval_dataloader
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)

        # Start the training
        start_time = time.time()
        for i in range(epochs):
            self.train()
            train_loss_bc, train_loss_domain, train_loss = 0.0, 0.0, 0.0
            for dirichlet_batch, right_batch, top_batch, domain_batch in zip(
                dirichlet_train_dl, right_train_dl, top_train_dl, domain_train_dl):
                loss_batch = self.trainBatch(dirichlet_batch, right_batch, top_batch, domain_batch, optimizer)
                train_loss_bc += loss_batch[0]
                train_loss_domain += loss_batch[1]
                train_loss += loss_batch[2]
            train_loss_bc /= len(dirichlet_train_dl) + len(right_train_dl) + len(top_train_dl)
            train_loss_domain /= len(domain_train_dl)
            train_loss /= len(dirichlet_train_dl) + len(right_train_dl) + len(top_train_dl) + len(domain_train_dl)
            
            # Save training metrics
            self.metrics['epochs'].append(starting_epoch + i)
            self.metrics['loss']['train']['bc'].append(train_loss_bc)
            self.metrics['loss']['train']['domain'].append(train_loss_domain)
            self.metrics['loss']['train']['total'].append(train_loss)

            # Evaluate the model
            self.eval()
            if eval_dataloader:
                loss_batch = self.computeLoss(dirichlet_eval_dl, right_eval_dl, top_eval_dl, domain_eval_dl)
                self.metrics['loss']['eval']['bc'].append(loss_batch[0])
                self.metrics['loss']['eval']['domain'].append(loss_batch[1])
                self.metrics['loss']['eval']['total'].append(loss_batch[2])
            
            # Print the progress
            if verbose and (i + 1) % epch_print == 0:
                eval_loss = loss_batch[2] if eval_dataloader else 'N/A'
                print(f"Epoch {i+1}/{epochs}: Loss ({train_loss:.4g}, {eval_loss:.4g})")

        self.metrics['time'] += time.time() - start_time

class LinearPoissonPINN(PoissonPINN):
    
    def __init__(
            self, 
            sizes: list, 
            activations: list,
            loss_fn = nn.MSELoss(),  
            init_type: str='xavier', 
            device: str='cpu',
    ):
        super(LinearPoissonPINN, self).__init__(sizes, activations, loss_fn, init_type, device)
    
    def trainBatch(self, bc_dirichlet_batch, domain_batch, optimizer):
        xy_dirichlet_batch, u_dirichlet_batch = bc_dirichlet_batch
        loss_bc  = self.computeSupervisedLoss(xy_dirichlet_batch, u_dirichlet_batch)
        loss_domain = self.computeResidualPDELoss(domain_batch)
        loss = loss_bc + loss_domain
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss_bc.item(), loss_domain.item(), loss.item()
    
    def computeSource(self, x, y, u):
        return torch.sin(pi*x)*torch.sin(pi*y)
    
    def computeLoss(self, bc_dirichlet_dl, domain_dl):
        # Compute dirichlet bc loss
        bc_loss = 0.0
        for xy_batch, u_batch in bc_dirichlet_dl:
            bc_loss += self.computeSupervisedLoss(xy_batch, u_batch).item()
        
        # Compute domain loss
        domain_loss = 0.0
        for xy_batch in domain_dl:
            domain_loss += self.computeResidualPDELoss(xy_batch).item()

        # Compute total loss
        total_loss = bc_loss + domain_loss

        # Normalize the losses
        bc_loss /= len(bc_dirichlet_dl)
        domain_loss /= len(domain_dl)
        total_loss /= len(bc_dirichlet_dl) + len(domain_dl)
        
        return bc_loss, domain_loss, total_loss
    
    def fit(self, train_dataloader, optimizer=optim.Adam, epochs=30, lr=1e-4, 
        regularization=0.0, eval_dataloader=None, verbose=True, epch_print=1):

        # Set the starting epoch
        last_epoch = self.metrics['epochs'][-1] if self.metrics['epochs'] else 0
        starting_epoch = last_epoch + 1
    
        # Get the dataloaders
        dirichlet_train_dl, domain_train_dl = train_dataloader
        dirichlet_eval_dl, domain_eval_dl = eval_dataloader
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)

        # Start the training
        start_time = time.time()
        for i in range(epochs):
            self.train()
            train_loss_bc, train_loss_domain, train_loss = 0.0, 0.0, 0.0
            for dirichlet_batch, domain_batch in zip(dirichlet_train_dl, domain_train_dl):
                loss_batch = self.trainBatch(dirichlet_batch, domain_batch, optimizer)
                train_loss_bc += loss_batch[0]
                train_loss_domain += loss_batch[1]
                train_loss += loss_batch[2]
            train_loss_bc /= len(dirichlet_train_dl)
            train_loss_domain /= len(domain_train_dl)
            train_loss /= len(dirichlet_train_dl) + len(domain_train_dl)

            # Save training metrics
            self.metrics['epochs'].append(starting_epoch + i)
            self.metrics['loss']['train']['bc'].append(train_loss_bc)
            self.metrics['loss']['train']['domain'].append(train_loss_domain)
            self.metrics['loss']['train']['total'].append(train_loss)
            
            # Evaluate the model
            self.eval()
            if eval_dataloader:
                loss_batch = self.computeLoss(dirichlet_eval_dl, domain_eval_dl)
                self.metrics['loss']['eval']['bc'].append(loss_batch[0])
                self.metrics['loss']['eval']['domain'].append(loss_batch[1])
                self.metrics['loss']['eval']['total'].append(loss_batch[2])
                
            # Print the progress
            if verbose and (i + 1) % epch_print == 0:
                eval_loss = loss_batch[2] if eval_dataloader else 'N/A'
                print(f"Epoch {i+1}/{epochs}: Loss ({train_loss:.4g}, {eval_loss:.4g})")

        self.metrics['time'] += time.time() - start_time