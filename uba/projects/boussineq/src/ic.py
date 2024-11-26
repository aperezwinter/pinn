import torch
import numpy as np
import pytorch_lightning as pl
from pyDOE import lhs
from torch.utils.data import DataLoader, Dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def gaussian(n: int=50, mu: float=0.5, sigma: float=0.05, a: float=0.05):
    # Gaussian function
    # t = 0, x in [0, 1]
    t = np.zeros((n, 1))
    x = lhs(1, n)
    tx = np.concatenate((t, x), axis=1)
    eta = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return tx, eta

def sine(n: int=50, a: float=1.0, b: float=1.0):
    # Sine function
    # t = 0, x in [0, 1]
    t = np.zeros((n, 1))
    x = lhs(1, n)
    tx = np.concatenate((t, x), axis=1)
    eta = a * np.sin(b * x)
    return tx, eta

class Supervised(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
class SupervisedDM(pl.LightningDataModule):
    def __init__(self, tx: np.ndarray, eta: np.ndarray, batch_size: int=32):
        super().__init__()
        self.n = tx.shape[0]
        self.tx = torch.tensor(tx, dtype=torch.float32, requires_grad=True, device=device)
        self.eta = torch.tensor(eta, dtype=torch.float32, device=device)
        self.batch_size = batch_size
    
    def setup(self, stage: str=None, percent: list=[0.7, 0.15]):
        warning = "Percentages must sum to less than 1.0 and be a list of two numbers."
        assert sum(percent) < 1.0 and len(percent) == 2, warning
        indices = np.random.permutation(self.n)
        idx_lim = [int(percent[0] * self.n), int(sum(percent) * self.n)]
        # Get the indices for the train, validation and test sets
        train_idx = indices[0:idx_lim[0]]   
        val_idx = indices[idx_lim[0]:idx_lim[1]]
        test_idx = indices[idx_lim[1]:]
        # Split the data into train, validation and test
        self.tx_train = self.tx[train_idx]
        self.tx_val = self.tx[val_idx]
        self.tx_test = self.tx[test_idx]
        # Split the labels into train, validation and test
        self.eta_train = self.eta[train_idx]
        self.eta_val = self.eta[val_idx]
        self.eta_test = self.eta[test_idx]
    
    def getTrainDataloader(self):
        train_split = Supervised(self.tx_train, self.eta_train)
        return DataLoader(train_split, batch_size=self.batch_size, shuffle=True)

    def getValDataloader(self):
        val_split = Supervised(self.tx_val, self.eta_val)
        return DataLoader(val_split, batch_size=self.batch_size)
    
    def getTestDataloader(self):
        test_split = Supervised(self.tx_test, self.eta_test)
        return DataLoader(test_split, batch_size=self.batch_size)
    
def getICDataLoader(n: int=1000, batch_size: int=32, ic: callable=gaussian):
    tx, eta = ic(n)
    dataset = SupervisedDM(tx, eta, batch_size=batch_size)
    dataset.setup()
    train_dl = dataset.getTrainDataloader()
    val_dl = dataset.getValDataloader()
    test_dl = dataset.getTestDataloader()
    return (train_dl, val_dl, test_dl)
