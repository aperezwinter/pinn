import torch
import numpy as np
from pyDOE import lhs
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def getCollocationPoints(n: int=50):
    return lhs(2, n)

class Unsupervised(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx]
    
class UnsupervisedDM(pl.LightningDataModule):
    def __init__(self, tx: np.ndarray, batch_size: int=32):
        super().__init__()
        self.n = tx.shape[0]
        self.tx = torch.tensor(tx, dtype=torch.float32, requires_grad=True, device=device)
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
    
    def getTrainDataloader(self):
        train_split = Unsupervised(self.tx_train)
        return DataLoader(train_split, batch_size=self.batch_size, shuffle=True)

    def getValDataloader(self):
        val_split = Unsupervised(self.tx_val)
        return DataLoader(val_split, batch_size=self.batch_size)
    
    def getTestDataloader(self):
        test_split = Unsupervised(self.tx_test)
        return DataLoader(test_split, batch_size=self.batch_size)
    
def getDomainDataLoader(n: int=1000, batch_size: int=32):
    tx = getCollocationPoints(n)
    dataset = UnsupervisedDM(tx, batch_size=batch_size)
    dataset.setup()
    train_dl = dataset.getTrainDataloader()
    val_dl = dataset.getValDataloader()
    test_dl = dataset.getTestDataloader()
    return (train_dl, val_dl, test_dl)
