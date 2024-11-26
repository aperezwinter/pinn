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

def getBC(n: int=50):
    # Boundary conditions
    # x = {0, 1}, for each boundary
    # t in [0, 1], so that t = 1 is the final condition
    t = lhs(1, n)
    x_left = np.zeros((n, 1))
    x_right = np.ones((n, 1))
    tx_left = np.concatenate((t, x_left), axis=1)
    tx_right = np.concatenate((t, x_right), axis=1)
    return tx_left, tx_right

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
    
def getBCDataLoader(n: int=1000, batch_size: int=32):
    tx_left, tx_right = getBC(n)
    # Left boundary condition
    dataset_left = UnsupervisedDM(tx_left, batch_size=batch_size)
    dataset_left.setup()
    train_dl_left = dataset_left.getTrainDataloader()
    val_dl_left = dataset_left.getValDataloader()
    test_dl_left = dataset_left.getTestDataloader()
    # Right boundary condition
    dataset_right = UnsupervisedDM(tx_right, batch_size=batch_size)
    dataset_right.setup()
    train_dl_right = dataset_right.getTrainDataloader()
    val_dl_right = dataset_right.getValDataloader()
    test_dl_right = dataset_right.getTestDataloader()
    # Assemble the dataloaders
    left_dl = (train_dl_left, val_dl_left, test_dl_left)
    right_dl = (train_dl_right, val_dl_right, test_dl_right)
    return left_dl, right_dl
