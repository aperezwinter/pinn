import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class Mesh(object):
    def __init__(self, a: float=1.0, b: float=1.0, nx: int=10, ny: int=10):
        self.a = a
        self.b = b
        self.nx = nx
        self.ny = ny
        self.hx = a / nx
        self.hy = b / ny
        self.x = np.linspace(0, a, nx + 1)
        self.y = np.linspace(0, b, ny + 1)
    
    def __getitem__(self, indices):
        """Allows access to mesh points using mesh[i, j] syntax."""
        i, j = indices
        return self.x[i], self.y[j]

    def size(self):
        """Returns the total size of the mesh."""
        return (self.nx + 1) * (self.ny + 1)

class LinearPoissonFDM(object):
    def __init__(
            self, 
            a: float=1.0, 
            b: float=1.0, 
            nx: int=10, 
            ny: int=10, 
    ):
        self.mesh = Mesh(a, b, nx, ny)
        self.u = np.zeros((self.mesh.nx + 1, self.mesh.ny + 1))
    
    def __getitem__(self, indices):
        """Allows access to solution points using [i, j] syntax."""
        return self.u[indices]
    
    def exact(self, x, y):
        """Exact solution."""
        return - np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)

    def error(self, norm: str='l2'):
        """Computes the error between the exact and numerical solutions."""
        u_exact = self.exact(self.mesh.x[:, None], self.mesh.y[None, :])
        error = (self.u - u_exact)
        if norm == 'l2':
            return np.linalg.norm(error, 2) / np.linalg.norm(u_exact, 2)
        elif norm == 'l1':
            return np.linalg.norm(error, 1) / np.linalg.norm(u_exact, 1)
        elif norm == 'max':
            return np.max(np.abs(error)) / np.max(np.abs(u_exact))
        else:
            raise ValueError(f"Invalid norm: {norm} is not supported.")
    
    def source(self, x, y):
        """Returns the source function for the differential equation."""
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def buildSystem(self, u):
        """Builds the system of equations to solve the differential equation."""
        nx, ny = self.mesh.nx, self.mesh.ny
        hx, hy = self.mesh.hx, self.mesh.hy
        n = (nx - 1) * (ny - 1)
        
        # Matrix A values at diagonal and adjacent positions
        main_diag = - 2 / hx**2 - 2 / hy**2
        off_diag_x = 1 / hx**2
        off_diag_y = 1 / hy**2
        
        diagonals = [
            main_diag * np.ones(n),
            off_diag_x * np.ones(n - 1),
            off_diag_x * np.ones(n - 1),
            off_diag_y * np.ones(n - (nx - 1)),
            off_diag_y * np.ones(n - (nx - 1))
        ]
        
        positions = [0, -1, 1, -(nx - 1), nx - 1]

        # Correct the adjacency matrix to avoid connections across rows
        for i in range(1, nx - 1):
            diagonals[1][i * (ny - 1) - 1] = 0  # Avoids wraparound in i+1
            diagonals[2][i * (ny - 1)] = 0      # Avoids wraparound in i-1

        A = diags(diagonals, positions, shape=(n, n)).tocsc()

        # Right-hand side vector
        x_interior = self.mesh.x[1:-1]
        y_interior = self.mesh.y[1:-1]
        X, Y = np.meshgrid(x_interior, y_interior, indexing='ij')
        b = self.source(X, Y).reshape(n)
        
        return A, b
    
    def solve(self):
        nx, ny = self.mesh.nx, self.mesh.ny
        u_interior = self.u[1:-1, 1:-1]
        A, b = self.buildSystem(self.u)
        u_interior = spsolve(A, b)
        self.u[1:-1, 1:-1] = u_interior.reshape((nx - 1, ny - 1))

    def plotExactSolution(self, figsize=(8, 5), save: bool=False, filename: str='exact_solution.png'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(self.mesh.x, self.mesh.y, indexing='ij')
        Z = self.exact(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title("Exact $u(x, y)$ - Linear Poisson Equation")
        plt.tight_layout()
        if save:
            plt.savefig(f'figures/{filename}', dpi=300, facecolor='w', edgecolor='w')
        plt.show()

    def plotNumericalSolution(self, figsize=(8, 5), save: bool=False, filename: str='numerical_solution.png'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(self.mesh.x, self.mesh.y, indexing='ij')
        Z = self.u
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title("Numerical $u(x, y)$ - Nonlinear Poisson Equation")
        plt.tight_layout()
        if save:
            plt.savefig(f'figures/{filename}', dpi=300, facecolor='w', edgecolor='w')
        plt.show()
    

class NonLinearPoissonFDM(object):
    def __init__(
            self, 
            a: float=1.0, 
            b: float=1.0, 
            nx: int=10, 
            ny: int=10, 
            tolerance: float=1e-4, 
            max_iter: int=5000, 
    ):
        self.mesh = Mesh(a, b, nx, ny)
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.u = np.zeros((self.mesh.nx + 1, self.mesh.ny + 1))
        
    def __getitem__(self, indices):
        """Allows access to solution points using [i, j] syntax."""
        return self.u[indices]
    
    def error(self, norm: str='l2'):
        """Computes the error between the exact and numerical solutions."""
        if norm == 'l2':
            return np.linalg.norm(self.u, 2)
        elif norm == 'l1':
            return np.linalg.norm(self.u, 1)
        elif norm == 'max':
            return np.max(np.abs(self.u))
        else:
            raise ValueError(f"Invalid norm: {norm} is not supported.")
    
    def source(self, u):
        """Returns the source function for the differential equation."""
        return 0.5 * np.exp(u)
    
    def sourceJacobian(self, u):
        """Jacobian of the source function."""
        return - 0.5 * np.exp(u)
    
    def buildSystem(self, u):
        nx, ny = self.mesh.nx, self.mesh.ny
        hx, hy = self.mesh.hx, self.mesh.hy
        n = (nx - 1) * (ny - 1)
        
        # Valores en la diagonal principal y las posiciones adyacentes de la matriz A
        main_diag = - 2 / hx**2 - 2 / hy**2
        off_diag_x = 1 / hx**2
        off_diag_y = 1 / hy**2
        
        diagonals = [
            main_diag * np.ones(n),
            off_diag_x * np.ones(n - 1),
            off_diag_x * np.ones(n - 1),
            off_diag_y * np.ones(n - (nx - 1)),
            off_diag_y * np.ones(n - (nx - 1))
        ]
        
        positions = [0, -1, 1, -(nx - 1), nx - 1]

        # Corregir la matriz de adyacencia para evitar conexiones incorrectas en filas
        for i in range(1, nx - 1):
            diagonals[1][i * (ny - 1) - 1] = 0  # Evita el wraparound en i+1
            diagonals[2][i * (ny - 1)] = 0      # Evita el wraparound en i-1

        A = diags(diagonals, positions, shape=(n, n)).tocsc()
        sJ = self.sourceJacobian(u[1:-1, 1:-1]).reshape(n)
        A = A + diags(sJ, 0).tocsc()

        # Vector del lado derecho
        u_interior = u[1:-1, 1:-1].reshape(n)
        F = u_interior - self.source(u_interior).reshape(n)

        # Condiciones de frontera Dirichlet y Neumann
        for j in range(1, ny):
            F[j - 1] = 0  # Dirichlet en y=0 (cara inferior)
            F[(nx - 2) * (ny - 1) + j - 1] -= off_diag_y * self.u[1, j]  # Neumann en y=1 (cara superior)

        for i in range(1, nx):
            F[(i - 1) * (ny - 1)] = 0  # Dirichlet en x=0 (cara izquierda)
            F[(i - 1) * (ny - 1) + ny - 2] -= off_diag_x * self.u[i, ny - 1]  # Neumann en x=1 (cara derecha)

        return A, F

    def solve(self):
        nx, ny = self.mesh.nx, self.mesh.ny
        u_interior = self.u[1:-1, 1:-1]

        for i in range(self.max_iter):
            A, F = self.buildSystem(self.u)
            delta_u = spsolve(A, -F).reshape((nx - 1, ny - 1))
            self.u[1:-1, 1:-1] = u_interior + delta_u
        
            error = 1 if i == 0 else np.linalg.norm(delta_u, ord=np.inf) / np.linalg.norm(u_interior, ord=np.inf)
            if error < self.tolerance:
                print(f"Convergencia lograda en {i + 1} iteraciones.")
                break
        else:
            print("Advertencia: Se alcanzó el número máximo de iteraciones sin convergencia.")

    def plotExactSolution(self, figsize=(8, 5), save: bool=False, filename: str='exact_solution.png'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(self.mesh.x, self.mesh.y, indexing='ij')
        Z = self.exact(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title("Exact $u(x, y)$")
        plt.tight_layout()
        if save:
            plt.savefig(f'figures/{filename}', dpi=300, facecolor='w', edgecolor='w')
        plt.show()

    def plotNumericalSolution(self, figsize=(8, 5), save: bool=False, filename: str='numerical_solution.png'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(self.mesh.x, self.mesh.y, indexing='ij')
        Z = self.u
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title("Numerical $u(x, y)$")
        plt.tight_layout()
        if save:
            plt.savefig(f'figures/{filename}', dpi=300, facecolor='w', edgecolor='w')
        plt.show()
        