import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
from fipy.tools import numerix
from fipy import CellVariable, Grid2D, DiffusionTerm, Viewer


class Domain(object):

    def __init__(self, Lx: float=1.0, Ly: float=1.0, nx: int=20, ny: int=20) -> None:
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.mesh = Grid2D(dx=self.dx, dy=self.dy, nx=nx, ny=ny)

    def getBoundsCoords(self):
        left = self.mesh.faceCenters[:, self.mesh.facesLeft]
        right = self.mesh.faceCenters[:, self.mesh.facesRight]
        top = self.mesh.faceCenters[:, self.mesh.facesTop]
        bottom = self.mesh.faceCenters[:, self.mesh.facesBottom]
        coords = np.hstack([left, right, top, bottom]).T
        return coords

    def getLeftBoundCoords(self):
        coords = self.mesh.faceCenters[:, self.mesh.facesLeft]
        coords = np.hstack([coords])
        return coords
    
    def getRightBoundCoords(self):
        coords = self.mesh.faceCenters[:, self.mesh.facesRight]
        coords = np.hstack([coords])
        return coords
    
    def getTopBoundCoords(self):
        coords = self.mesh.faceCenters[:, self.mesh.facesTop]
        coords = np.hstack([coords])
        return coords
    
    def getBottomBoundCoords(self):
        coords = self.mesh.faceCenters[:, self.mesh.facesBottom]
        coords = np.hstack([coords])
        return coords
    
    def getDomainCoords(self):
        return self.mesh.cellCenters.value.T
    
    def getCoords(self):
        bounds = self.getBoundsCoords()
        domain = self.getDomainCoords()
        return np.vstack([bounds, domain])
    
    def getNumberOfPoints(self):
        return (self.nx + 1) * (self.ny + 1)
    
    def getNumberOfCells(self):
        return self.nx * self.ny


class LinearPoisson(object):

    def __init__(self, Lx: float=1.0, Ly: float=1.0, nx: int=20, ny: int=20) -> None:
        self.domain = Domain(Lx, Ly, nx, ny)
        self.k = 1
        self.u = CellVariable(mesh=self.domain.mesh, name="u")
    
    def computeErrorL2(self, values: np.ndarray=None):
        u_num = self.getSolution()
        if values:
            values_norm = np.linalg.norm(values, 2)
            error = np.linalg.norm(values - u_num) / values_norm
        else: 
            u_exact = self.getExactSolution()
            u_exact_norm = np.linalg.norm(u_exact, 2)
            error = np.linalg.norm(u_exact - u_num) / u_exact_norm
        return error
    
    def computeErrorL2InPoints(self, points: np.ndarray, method: str, values: np.ndarray=None):
        u_interpolated = self.interpolate(points, method)
        if values:
            values_norm = np.linalg.norm(values, 2)
            error = np.linalg.norm(u_interpolated - values) / values_norm
        else:
            u_exact = self.computeExactSolution(points[:, 0], points[:, 1])
            u_exact_norm = np.linalg.norm(u_exact, 2)
            error = np.linalg.norm(u_exact - u_interpolated) / u_exact_norm
        return error

    def computeExactSolution(self, x, y):
        return - np.sin(pi * x) * np.sin(pi * y) / (2 * pi**2)
    
    def computeNormL2(self):
        u_num = self.getSolution()
        return np.linalg.norm(u_num)
    
    def computeNormL2InPoints(self, points: np.ndarray, method: str):
        u_interpolated = self.interpolate(points, method)
        return np.linalg.norm(u_interpolated)

    def getExactSolution(self):
        coords = self.domain.getCoords()
        x, y = coords[:, 0], coords[:, 1]
        u = self.computeExactSolution(x, y)
        return u
    
    def getSolution(self):
        u_bound = self.getSolutionInBoundary()
        u_values = np.hstack([u_bound, self.u.value])
        return u_values
    
    def getSolutionInBoundary(self):
        u_left = self.u.arithmeticFaceValue[self.domain.mesh.facesLeft]
        u_right = self.u.arithmeticFaceValue[self.domain.mesh.facesRight]
        u_top = self.u.arithmeticFaceValue[self.domain.mesh.facesTop]
        u_bottom = self.u.arithmeticFaceValue[self.domain.mesh.facesBottom]
        return np.hstack([u_left, u_right, u_top, u_bottom])
    
    def interpolate(self, points: np.ndarray, method: str="linear"):
        assert method in ["linear", "nearest", "cubic"], f"{method} interpolation's method not allowed."
        coords = self.domain.getCoords()
        u_bound = self.getSolutionInBoundary()
        u_values = np.hstack([u_bound, self.u.value])
        u_interp_values = griddata(points=coords, values=u_values, xi=points, 
                                   method=method, fill_value=0.0)
        return u_interp_values
    
    def plotSolution(self, figsize=(6, 4), save: bool=False, 
                     filename: str="nonlinear_solution.png", 
                     title: str="Linear Poisson - Grid"):
        # Sample data
        r = self.domain.getCoords()
        x, y = r[:,0], r[:,1]
        u = self.getSolution()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # 3D surface plot
        min_u, max_u = u.min(), u.max()
        levels = np.linspace(min_u, max_u, 10)
        ax.plot_trisurf(x, y, u, cmap='viridis', antialiased=False, edgecolor='none')
        ax.tricontour(x, y, u, levels=levels, linewidths = 1, colors="black", offset=min_u)
        ax.tricontourf(x, y, u, levels=levels, cmap='viridis', offset=min_u, alpha=0.75)
        
        # Z-axis limits
        ax.set_zlim(min_u, max_u)
        ax.set_box_aspect(None, zoom=0.8)

        # Axis labels
        ax.set_xlabel(r'$x$', labelpad=5)
        ax.set_ylabel(r'$y$', labelpad=5)
        ax.set_zlabel(r'$u$', labelpad=10)

        plt.tight_layout()
        plt.title(title)
        
        # Save
        if save:
            plt.savefig(filename, dpi=200, facecolor='w', edgecolor='w')
            plt.close()
        else:
            plt.show()

    def plotSolutionInPoints(self, points: np.ndarray, method: str="linear", figsize=(8, 5), 
                             save: bool=False, filename: str='nonlinear_solution.png', 
                             title: str="Linear Poisson - Interpolated"):
        # Sample data
        x, y = points[:, 0], points[:, 1]
        u = self.interpolate(points, method)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # 3D surface plot
        min_u, max_u = u.min(), u.max()
        levels = np.linspace(min_u, max_u, 10)
        ax.plot_trisurf(x, y, u, cmap='viridis', antialiased=False, edgecolor='none')
        ax.tricontour(x, y, u, levels=levels, linewidths = 1, colors="black", offset=min_u)
        ax.tricontourf(x, y, u, levels=levels, cmap='viridis', offset=min_u, alpha=0.75)
        
        # Z-axis limits
        ax.set_zlim(min_u, max_u)
        ax.set_box_aspect(None, zoom=0.8)

        # Axis labels
        ax.set_xlabel(r'$x$', labelpad=5)
        ax.set_ylabel(r'$y$', labelpad=5)
        ax.set_zlabel(r'$u$', labelpad=10)

        plt.tight_layout()
        plt.title(title)
        
        # Save
        if save:
            plt.savefig(filename, dpi=200, facecolor='w', edgecolor='w')
            plt.close()
        else:
            plt.show()
    
    def source(self, x, y):
        return numerix.sin(pi * x) * numerix.sin(pi * y)
    
    def solve(self, tol: float=1e-4):
        x = self.domain.mesh.cellCenters[0]
        y = self.domain.mesh.cellCenters[1]

        # Define PDE
        pde = DiffusionTerm(coeff=self.k) - self.source(x, y)

        # Set boundary conditions (Dirichlet homogeneous)
        self.u.constrain(0, self.domain.mesh.facesTop)
        self.u.constrain(0, self.domain.mesh.facesBottom)
        self.u.constrain(0, self.domain.mesh.facesLeft)
        self.u.constrain(0, self.domain.mesh.facesRight)
        
        # Solve partial differential equation
        while pde.sweep(var=self.u) > tol:
            pass


class NonLinearPoisson(object):

    def __init__(self, Lx: float=1.0, Ly: float=1.0, nx: int=20, ny: int=20) -> None:
        self.domain = Domain(Lx, Ly, nx, ny)
        self.k = 1
        self.u = CellVariable(mesh=self.domain.mesh, name="u")

    def computeErrorL2(self, values: np.ndarray):
        u_num = self.getSolution()
        values_norm = np.linalg.norm(values, 2)
        error = np.linalg.norm(values - u_num) / values_norm
        return error
    
    def computeErrorL2InPoints(self, points: np.ndarray, values: np.ndarray, method: str="linear"):
        u_interpolated = self.interpolate(points, method)
        values_norm = np.linalg.norm(values, 2)
        error = np.linalg.norm(u_interpolated - values) / values_norm
        return error
    
    def computeNormL2(self):
        u_num = self.getSolution()
        return np.linalg.norm(u_num)
    
    def computeNormL2InPoints(self, points: np.ndarray, method: str):
        u_interpolated = self.interpolate(points, method)
        return np.linalg.norm(u_interpolated)
    
    def getSolution(self):
        u_bound = self.getSolutionInBoundary()
        u_values = np.hstack([u_bound, self.u.value])
        return u_values
    
    def getSolutionInBoundary(self):
        u_left = self.u.arithmeticFaceValue[self.domain.mesh.facesLeft]
        u_right = self.u.arithmeticFaceValue[self.domain.mesh.facesRight]
        u_top = self.u.arithmeticFaceValue[self.domain.mesh.facesTop]
        u_bottom = self.u.arithmeticFaceValue[self.domain.mesh.facesBottom]
        return np.hstack([u_left, u_right, u_top, u_bottom])
    
    def interpolate(self, points: np.ndarray, method: str="linear"):
        assert method in ["linear", "nearest", "cubic"], f"{method} interpolation's method not allowed."
        coords = self.domain.getCoords()
        u_bound = self.getSolutionInBoundary()
        u_values = np.hstack([u_bound, self.u.value])
        u_interp_values = griddata(points=coords, values=u_values, xi=points, 
                                   method=method, fill_value=0.0)
        return u_interp_values
    
    def plotSolution(self, figsize=(6, 4), save: bool=False, 
                     filename: str="nonlinear_solution.png", 
                     title: str="Non Linear Poisson - Grid"):
        # Sample data
        r = self.domain.getCoords()
        x, y = r[:,0], r[:,1]
        u = self.getSolution()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        
        # 3D surface plot
        min_u, max_u = u.min(), u.max()
        levels = np.linspace(min_u, max_u, 10)
        ax.plot_trisurf(x, y, u, cmap='viridis', antialiased=False, edgecolor='none')
        ax.tricontour(x, y, u, levels=levels, linewidths = 1, colors="black", offset=min_u)
        ax.tricontourf(x, y, u, levels=levels, cmap='viridis', offset=min_u, alpha=0.75)
        
        # Z-axis limits
        ax.set_zlim(min_u, max_u)
        ax.set_box_aspect(None, zoom=0.9)

        # Axis labels
        ax.set_xlabel(r'$x$', labelpad=5)
        ax.set_ylabel(r'$y$', labelpad=5)
        ax.set_zlabel(r'$u$', labelpad=10)

        plt.title(title)
        plt.tight_layout()
        
        # Save
        if save:
            plt.savefig(filename, dpi=100, facecolor='w', edgecolor='w')
            plt.close()
        else:
            plt.show()

    def plotSolutionInPoints(self, points: np.ndarray, method: str="linear", figsize=(8, 5), 
                             save: bool=False, filename: str='nonlinear_solution.png', 
                             title: str="Non Linear Poisson - Interpolated"):
        # Sample data
        x, y = points[:, 0], points[:, 1]
        u = self.interpolate(points, method)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        
        # 3D surface plot
        min_u, max_u = u.min(), u.max()
        levels = np.linspace(min_u, max_u, 10)
        ax.plot_trisurf(x, y, u, cmap='viridis', antialiased=False, edgecolor='none')
        ax.tricontour(x, y, u, levels=levels, linewidths = 1, colors="black", offset=min_u)
        ax.tricontourf(x, y, u, levels=levels, cmap='viridis', offset=min_u, alpha=0.75)
        
        # Z-axis limits
        ax.set_zlim(min_u, max_u)
        ax.set_box_aspect(None, zoom=0.8)

        # Axis labels
        ax.set_xlabel(r'$x$', labelpad=5)
        ax.set_ylabel(r'$y$', labelpad=5)
        ax.set_zlabel(r'$u$', labelpad=10)

        plt.tight_layout()
        plt.title(title)
        
        # Save
        if save:
            plt.savefig(filename, dpi=100, facecolor='w', edgecolor='w')
            plt.close()
        else:
            plt.show()
    
    def source(self, u):
        return 0.5 * numerix.exp(u)
    
    def solve(self, tol: float=1e-4):
        # Define PDE
        pde = DiffusionTerm(coeff=self.k) - self.source(self.u)

        # Set boundary conditions
        # Bottom, Left -> u = 0
        # Top, Right -> du = 0 (by default)
        self.u.constrain(0, self.domain.mesh.facesBottom)
        self.u.constrain(0, self.domain.mesh.facesLeft)
        
        # Solve partial differential equation
        while pde.sweep(var=self.u) > tol:
            pass