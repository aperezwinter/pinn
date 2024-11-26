import time, json, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(
            self, 
            sizes: list, 
            activations: list,
            loss_fn = nn.MSELoss(),  
            init_type: str='xavier', 
            device: str='cpu',
            dropout: float=0.2,
            loss_tol: float=1e-12
    ):
        super(PINN, self).__init__()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.loss_fn = loss_fn
        self.dropout = dropout
        self.device = device
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0
        self.lambda_domain = 1.0
        self.loss_tol = loss_tol
        
        # Define the layers.
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(activations):
                layers.append(activations[i])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

        # Initialize the parameters.
        self.initializeWeights(init_type)

        # Define the metrics.
        self.metrics = {
            'epochs': [], 
            'loss': {
                'train': {'ic': [], 'bc': [], 'domain': [], 'total': []}, 
                'val'  : {'ic': [], 'bc': [], 'domain': [], 'total': []},
                'test' : {'ic': None, 'bc': None, 'domain': None, 'total': None}
            }, 
            'time': 0.0
        }

    def fit(self, train_dataloader, optimizer=optim.Adam, epochs=30, lr=1e-4, 
        regularization=0.0, val_dataloader=None, verbose=True, epoch_print=1,
        weighted_loss: bool=False, alpha: float=0.9, epoch_refresh: int=10):

        # Set the starting epoch
        last_epoch = self.metrics['epochs'][-1] if self.metrics['epochs'] else 0
        starting_epoch = last_epoch + 1
    
        # Get the dataloaders
        ic_train_dl, left_train_dl, right_train_dl, domain_train_dl = train_dataloader
        ic_val_dl, left_val_dl, right_val_dl, domain_val_dl = val_dataloader
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)

        # Start the training
        start_time = time.time()
        for i in range(epochs):
            # Train the model
            self.train()
            for ic_batch, left_batch, right_batch, domain_batch in zip(
                ic_train_dl, left_train_dl, right_train_dl, domain_train_dl):
                self.trainBatch(ic_batch, left_batch, right_batch, domain_batch, optimizer)
            
            # Evaluate the model
            self.eval()
            self.metrics['epochs'].append(starting_epoch + i)
            train_loss = self.computeLoss(ic_train_dl, left_train_dl, right_train_dl, domain_train_dl)
            self.metrics['loss']['train']['ic'].append(train_loss[0].item())
            self.metrics['loss']['train']['bc'].append(train_loss[1].item())
            self.metrics['loss']['train']['domain'].append(train_loss[2].item())
            self.metrics['loss']['train']['total'].append(train_loss[3].item())
            if val_dataloader:
                val_loss = self.computeLoss(ic_val_dl, left_val_dl, right_val_dl, domain_val_dl)
                self.metrics['loss']['val']['ic'].append(val_loss[0].item())
                self.metrics['loss']['val']['bc'].append(val_loss[1].item())
                self.metrics['loss']['val']['domain'].append(val_loss[2].item())
                self.metrics['loss']['val']['total'].append(val_loss[3].item())
            
            # Refresh lambda
            if weighted_loss and (i + 1) % epoch_refresh == 0:
                self.train()
                lambda_loss = self.computeLossForLambda(ic_train_dl, left_train_dl, right_train_dl, domain_train_dl)
                self.computeLambda(lambda_loss[0], lambda_loss[1], lambda_loss[2], optimizer, alpha)
            
            # Print the progress
            if verbose and (i + 1) % epoch_print == 0:
                val_loss = val_loss[3].item() if val_dataloader else 'N/A'
                text = f"Epoch {starting_epoch + i}/{starting_epoch + epochs}: "
                text += f"Loss ({train_loss[3].item():.4g}, {val_loss:.4g})"
                print(text)

        self.metrics['time'] += time.time() - start_time
    
    def test(self, test_dataloader):
        self.eval()
        ic_loss, bc_loss, domain_loss, total_loss = self.computeLoss(test_dataloader)
        self.metrics['loss']['test']['ic'] = ic_loss.item()
        self.metrics['loss']['test']['bc'] = bc_loss.item()
        self.metrics['loss']['test']['domain'] = domain_loss.item()
        self.metrics['loss']['test']['total'] = total_loss.item()
    
    def forward(self, tx):
        return self.layers(tx)
    
    def predict(self, tx):
        self.eval()
        return self.forward(tx)
    
    def initializeWeights(self, init_type):
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
    
    def initializeLambda(self, ic: float=1.0, bc: float=1.0, domain: float=1.0):
        self.lambda_ic = ic
        self.lambda_bc = bc
        self.lambda_domain = domain

    def computeLambda(self, loss_ic: float, loss_bc: float, loss_domain: float, optimizer: optim.Optimizer, alpha: float=0.9):
        # Compute gradient norm of the parameters 
        ## due to the initial condition loss
        optimizer.zero_grad()
        loss_ic.backward(retain_graph=True)
        params_grad_norm_ic = self.computeParamsGradNorm()
        ## due to the boundary condition loss
        optimizer.zero_grad()
        loss_bc.backward(retain_graph=True)
        params_grad_norm_bc = self.computeParamsGradNorm()
        ## due to the domain loss
        optimizer.zero_grad()
        loss_domain.backward(retain_graph=True)
        params_grad_norm_domain = self.computeParamsGradNorm()

        # Compute new lambdas
        total_grad_norm = params_grad_norm_ic + params_grad_norm_bc + params_grad_norm_domain
        new_lambda_ic = total_grad_norm / params_grad_norm_ic
        new_lambda_bc = total_grad_norm / params_grad_norm_bc
        new_lambda_domain = total_grad_norm / params_grad_norm_domain

        # Update lambda
        self.lambda_ic = alpha * self.lambda_ic + (1 - alpha) * new_lambda_ic.item()
        self.lambda_bc = alpha * self.lambda_bc + (1 - alpha) * new_lambda_bc.item()
        self.lambda_domain = alpha * self.lambda_domain + (1 - alpha) * new_lambda_domain.item()

    def computeLossForLambda(self, ic_dl, left_dl, right_dl, domain_dl):        
        # Compute initial condition loss
        ic_loss = torch.tensor(0.0, device=self.device)
        for tx_batch, eta_batch in ic_dl:
            ic_loss += self.computeSupervisedLoss(tx_batch, eta_batch)
        
        # Compute bc loss
        bc_loss = torch.tensor(0.0, device=self.device)
        for left_batch, right_batch in zip(left_dl, right_dl):
            bc_loss += self.computeBCsLoss(left_batch, right_batch)
        
        # Compute domain loss
        domain_loss = torch.tensor(0.0, device=self.device)
        for tx_batch in domain_dl:
            domain_loss += self.computePDELoss(tx_batch)

        # Set the loss to tolerance if it is too small
        if ic_loss.item() < self.loss_tol:
            ic_loss = torch.tensor(self.loss_tol, dtype=torch.float32, device=self.device, requires_grad=True)
        if bc_loss.item() < self.loss_tol:
            bc_loss = torch.tensor(self.loss_tol, dtype=torch.float32, device=self.device, requires_grad=True)
        if domain_loss.item() < self.loss_tol:
            domain_loss = torch.tensor(self.loss_tol, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Compute total loss
        lambda_ic = torch.tensor(self.lambda_ic, dtype=torch.float32, device=self.device)
        lambda_bc = torch.tensor(self.lambda_bc, dtype=torch.float32, device=self.device)
        lambda_domain = torch.tensor(self.lambda_domain, dtype=torch.float32, device=self.device)
        total_loss = ic_loss * lambda_ic
        total_loss += bc_loss * lambda_bc
        total_loss += domain_loss * lambda_domain
        
        return ic_loss, bc_loss, domain_loss, total_loss
    
    def computeLoss(self, ic_dl, left_dl, right_dl, domain_dl):
        # Compute initial condition loss
        ic_loss = 0.0
        for tx_batch, eta_batch in ic_dl:
            ic_loss += self.computeSupervisedLoss(tx_batch, eta_batch)

        # Compute bc loss
        bc_loss = 0.0
        for left_batch, right_batch in zip(left_dl, right_dl):
            bc_loss += self.computeBCsLoss(left_batch, right_batch)

        # Compute domain loss
        domain_loss = 0.0
        for tx_batch in domain_dl:
            domain_loss += self.computePDELoss(tx_batch)

        # Set the loss to tolerance if it is too small
        ic_loss = torch.tensor(self.loss_tol, device=self.device) if ic_loss.item() < self.loss_tol else ic_loss
        bc_loss = torch.tensor(self.loss_tol, device=self.device) if bc_loss.item() < self.loss_tol else bc_loss
        domain_loss = torch.tensor(self.loss_tol, device=self.device) if domain_loss.item() < self.loss_tol else domain_loss

        # Compute total loss
        total_loss = ic_loss * self.lambda_ic
        total_loss += bc_loss * self.lambda_bc
        total_loss += domain_loss * self.lambda_domain
        
        return ic_loss, bc_loss, domain_loss, total_loss
    
    def computeSupervisedLoss(self, input_batch, target_batch, gradient: bool=False, axis: int=0):
        prediction_batch = self.forward(input_batch)
        if gradient:
            grad_batch = torch.autograd.grad(prediction_batch, input_batch, torch.ones_like(prediction_batch))[0]
            grad_batch = grad_batch[:, axis:axis+1] # Get the gradient in the specified axis
            loss = self.loss_fn(grad_batch, target_batch)
        else:
            loss = self.loss_fn(prediction_batch, target_batch)
        return loss
    
    def computeBCsLoss(self, tx_left_batch, tx_right_batch):
        eta_left = self.forward(tx_left_batch)
        eta_right = self.forward(tx_right_batch)

        # Compute 1st order derivatives
        eta_left_tx = torch.autograd.grad(eta_left, tx_left_batch, torch.ones_like(eta_left), create_graph=True)[0]
        eta_right_tx = torch.autograd.grad(eta_right, tx_right_batch, torch.ones_like(eta_right), create_graph=True)[0]
        eta_left_t = eta_left_tx[:, 0:1]
        eta_left_x = eta_left_tx[:, 1:2]
        eta_right_t = eta_right_tx[:, 0:1]
        eta_right_x = eta_right_tx[:, 1:2]

        # Compute the boundary loss
        loss = self.loss_fn(eta_left_t, eta_left_x)
        loss += self.loss_fn(eta_right_t, -1 * eta_right_x)
        return loss
    
    def computePDELoss(self, tx_batch):
        eta = self.forward(tx_batch)
        eta2 = eta ** 2
        eta_tx = torch.autograd.grad(eta, tx_batch, torch.ones_like(eta), create_graph=True)[0]
        eta_t, eta_x = eta_tx[:, 0:1], eta_tx[:, 1:2]
        eta_tt = torch.autograd.grad(eta_t, tx_batch, torch.ones_like(eta_t), create_graph=True)[0][:, 0:1]
        eta_xx = torch.autograd.grad(eta_x, tx_batch, torch.ones_like(eta_x), create_graph=True)[0][:, 1:2]

        aux = 3 * eta2 + eta_xx
        aux_tx = torch.autograd.grad(aux, tx_batch, torch.ones_like(aux), create_graph=True)[0]
        aux_x = aux_tx[:, 1:2]
        aux_xx = torch.autograd.grad(aux_x, tx_batch, torch.ones_like(aux_x), create_graph=True)[0][:, 1:2]

        f = eta_tt - eta_xx - aux_xx                    # compute residual
        return self.loss_fn(f, torch.zeros_like(f))     # compute loss
    
    def computeParamsNorm(self):
        norms = [p.norm() ** 2 for p in self.parameters() if p.requires_grad]
        return torch.sqrt(torch.sum(torch.stack(norms)))
    
    def computeParamsGradNorm(self):
        grads = [p.grad.norm() ** 2 for p in self.parameters() if p.requires_grad and p.grad is not None]
        return torch.sqrt(torch.sum(torch.stack(grads)))
    
    def trainBatch(self, ic_batch, left_batch, right_batch, domain_batch, optimizer):
        ic_tx, ic_eta = ic_batch
        loss_ic = self.computeSupervisedLoss(ic_tx, ic_eta)
        loss_bc = self.computeBCsLoss(left_batch, right_batch)
        loss_domain = self.computePDELoss(domain_batch)
        loss = loss_ic * self.lambda_ic 
        loss += loss_bc * self.lambda_bc
        loss += loss_domain * self.lambda_domain
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss_ic.item(), loss_bc.item(), loss_domain.item(), loss.item()
    
    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, model_path: str="model_params.pth", metrics_path: str="metrics.txt"):
        torch.save(self.state_dict(), model_path)
        with open(metrics_path, 'w') as f:
            f.truncate()
            json.dump(self.metrics, f)
        f.close()

    def load(self, model_path: str="model_params.pth", metrics_path: str="metrics.txt"):
        self.load_state_dict(torch.load(model_path, weights_only=True))
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
        f.close()
        self.to(self.device)

    def plotLoss(self, figsize=(10, 5), save: bool=False, filename: str="loss.png", log_x: bool=False):
        # Plot 4 subplots: train and val loss for ic, bc, domain and total (2 x 2)
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        # Plot ic loss
        axs[0, 0].plot(self.metrics['epochs'], self.metrics['loss']['train']['ic'], label='Train')
        axs[0, 0].plot(self.metrics['epochs'], self.metrics['loss']['val']['ic'], label='Val')
        axs[0, 0].set_title('IC')
        # Plot bc loss
        axs[0, 1].plot(self.metrics['epochs'], self.metrics['loss']['train']['bc'], label='Train')
        axs[0, 1].plot(self.metrics['epochs'], self.metrics['loss']['val']['bc'], label='Val')
        axs[0, 1].set_title('BC')
        # Plot domain loss
        axs[1, 0].plot(self.metrics['epochs'], self.metrics['loss']['train']['domain'], label='Train')
        axs[1, 0].plot(self.metrics['epochs'], self.metrics['loss']['val']['domain'], label='Val')
        axs[1, 0].set_title('Domain')
        # Plot total loss
        axs[1, 1].plot(self.metrics['epochs'], self.metrics['loss']['train']['total'], label='Train')
        axs[1, 1].plot(self.metrics['epochs'], self.metrics['loss']['val']['total'], label='Val')
        axs[1, 1].set_title('Total')
        # Set axes properties
        for ax in axs.flatten():
            ax.set_yscale('log')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend(loc='best')
            ax.grid(True, which='both', ls='--', alpha=0.5, c='gray')
        if log_x:
            for ax in axs.flatten():
                ax.set_xscale('log')
        # Save the figure
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w')
        plt.close()