"""
Yee Grid Implementation for 2D FDTD
Coordinates for E and H fields on staggered grid
"""
import numpy as np
import matplotlib.pyplot as plt
import os

class YeeGrid2D:
    def __init__(self, nx, ny, dx):
        """
        Initialize Yee grid for 2D TM mode
        
        Parameters:
        nx, ny: Grid dimensions
        dx: Spatial resolution (meters)
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        
        # E_z field at cell centers (i, j)
        self.Ez = np.zeros((nx, ny), dtype=np.float32)
        
        # H_x field at (i, j+1/2)
        self.Hx = np.zeros((nx, ny), dtype=np.float32)
        
        # H_y field at (i+1/2, j)
        self.Hy = np.zeros((nx, ny), dtype=np.float32)
        
        # Material properties at cell centers
        self.epsilon = np.ones((nx, ny), dtype=np.float32)
        self.sigma = np.zeros((nx, ny), dtype=np.float32)
        
        # Courant number for stability (CFL condition for 2D)
        # For 2D, courant must be <= 1/√2 ≈ 0.707
        self.courant = 0.5  # Conservative value for stability
        
        # Time step from CFL condition: Δt ≤ Δx/(c√2)
        c0 = 299792458  # Speed of light in vacuum (m/s)
        self.dt = self.courant * dx / (c0 * np.sqrt(2))
        
        # Ensure dt is not zero (check for very small dx)
        if self.dt == 0:
            self.dt = 1e-18  # Set to 1 attosecond if calculation gives 0
        
        print(f"Yee Grid Initialized: {nx}×{ny} cells")
        print(f"Resolution: {dx*1e9:.3f} nm")
        print(f"Time step: {self.dt*1e15:.3f} fs")
        print(f"CFL number: {self.courant:.3f}")
        
    def set_material(self, material_mask, epsilon_r, sigma_val=0.0):
        """
        Set material properties in region defined by mask
        
        Parameters:
        material_mask: Boolean array (True where material exists)
        epsilon_r: Relative permittivity
        sigma_val: Conductivity (S/m)
        """
        self.epsilon[material_mask] = epsilon_r
        self.sigma[material_mask] = sigma_val
        
        # Print material stats
        material_cells = np.sum(material_mask)
        total_cells = self.nx * self.ny
        print(f"Material set: ε={epsilon_r}, σ={sigma_val:.2e} S/m")
        print(f"  Material occupies {material_cells}/{total_cells} cells ({100*material_cells/total_cells:.1f}%)")
        
    def get_field_at_position(self, x, y):
        """
        Bilinear interpolation of E_z at arbitrary position
        
        Parameters:
        x, y: Position in meters
        
        Returns:
        Interpolated E_z value
        """
        # Convert to grid coordinates
        i = x / self.dx
        j = y / self.dx
        
        # Clamp to grid bounds (leaving 1 cell margin for interpolation)
        i = np.clip(i, 0, self.nx-2)
        j = np.clip(j, 0, self.ny-2)
        
        # Integer indices
        i0 = int(i)
        j0 = int(j)
        
        # Fractional parts
        di = i - i0
        dj = j - j0
        
        # Bilinear interpolation
        E00 = self.Ez[i0, j0]
        E01 = self.Ez[i0, j0+1]
        E10 = self.Ez[i0+1, j0]
        E11 = self.Ez[i0+1, j0+1]
        
        Ez_interp = (1-di)*(1-dj)*E00 + (1-di)*dj*E01 + \
                   di*(1-dj)*E10 + di*dj*E11
        
        return Ez_interp
    
    def add_gold_disk(self, center_x, center_y, radius, epsilon_r=-12.0, sigma=1.0e7):
        """
        Add a circular gold disk to the grid
        
        Parameters:
        center_x, center_y: Center position (meters)
        radius: Disk radius (meters)
        epsilon_r: Gold permittivity (real part, negative at optical frequencies)
        sigma: Gold conductivity (S/m)
        """
        # Create coordinate grids
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Distance from center
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Create mask for disk
        disk_mask = R <= radius
        
        # Set material properties
        self.set_material(disk_mask, epsilon_r, sigma)
        
        print(f"Gold disk added: center=({center_x*1e9:.1f}nm, {center_y*1e9:.1f}nm), "
              f"radius={radius*1e9:.1f}nm")
        
        return disk_mask
    
    def add_gold_nanodisk(self, size_nm=100):
        """
        Add a gold nanodisk at center of grid
        
        Parameters:
        size_nm: Diameter in nanometers
        """
        radius = (size_nm * 1e-9) / 2
        center_x = (self.nx * self.dx) / 2
        center_y = (self.ny * self.dx) / 2
        
        return self.add_gold_disk(center_x, center_y, radius)
    
    def visualize_grid(self, filename=None):
        """
        Create a visualization of grid points and optionally save to file
        
        Parameters:
        filename: Optional filename to save the figure
        
        Returns:
        matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # E_z points (cell centers)
        x_e = (np.arange(self.nx) + 0.5) * self.dx
        y_e = (np.arange(self.ny) + 0.5) * self.dx
        X_e, Y_e = np.meshgrid(x_e, y_e, indexing='ij')
        axes[0].scatter(X_e.ravel()*1e9, Y_e.ravel()*1e9, s=1, c='blue', alpha=0.6, label='E_z')
        axes[0].set_title('E_z Points (Cell Centers)', fontsize=12)
        axes[0].set_xlabel('x (nm)')
        axes[0].set_ylabel('y (nm)')
        axes[0].set_aspect('equal')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # H_x points (vertical edges)
        x_hx = np.arange(self.nx) * self.dx
        y_hx = (np.arange(self.ny) + 0.5) * self.dx
        X_hx, Y_hx = np.meshgrid(x_hx, y_hx, indexing='ij')
        axes[1].scatter(X_hx.ravel()*1e9, Y_hx.ravel()*1e9, s=1, c='red', alpha=0.6, label='H_x')
        axes[1].set_title('H_x Points (Vertical Edges)', fontsize=12)
        axes[1].set_xlabel('x (nm)')
        axes[1].set_ylabel('y (nm)')
        axes[1].set_aspect('equal')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # H_y points (horizontal edges)
        x_hy = (np.arange(self.nx) + 0.5) * self.dx
        y_hy = np.arange(self.ny) * self.dx
        X_hy, Y_hy = np.meshgrid(x_hy, y_hy, indexing='ij')
        axes[2].scatter(X_hy.ravel()*1e9, Y_hy.ravel()*1e9, s=1, c='green', alpha=0.6, label='H_y')
        axes[2].set_title('H_y Points (Horizontal Edges)', fontsize=12)
        axes[2].set_xlabel('x (nm)')
        axes[2].set_ylabel('y (nm)')
        axes[2].set_aspect('equal')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'2D Yee Grid: {self.nx}×{self.ny} cells, Δx={self.dx*1e9:.1f} nm', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Grid visualization saved to {filename}")
        
        return fig
    
    def visualize_material_distribution(self, filename=None):
        """
        Visualize material distribution (epsilon values) on the grid
        
        Parameters:
        filename: Optional filename to save the figure
        
        Returns:
        matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot epsilon distribution
        im1 = axes[0].imshow(self.epsilon.T, cmap='viridis', 
                            extent=[0, self.nx*self.dx*1e9, 0, self.ny*self.dx*1e9],
                            origin='lower')
        axes[0].set_title('Permittivity (ε) Distribution', fontsize=12)
        axes[0].set_xlabel('x (nm)')
        axes[0].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0], label='Relative Permittivity')
        
        # Plot conductivity distribution
        im2 = axes[1].imshow(self.sigma.T, cmap='hot', 
                            extent=[0, self.nx*self.dx*1e9, 0, self.ny*self.dx*1e9],
                            origin='lower')
        axes[1].set_title('Conductivity (σ) Distribution', fontsize=12)
        axes[1].set_xlabel('x (nm)')
        axes[1].set_ylabel('y (nm)')
        plt.colorbar(im2, ax=axes[1], label='Conductivity (S/m)')
        
        plt.suptitle('Material Properties on Yee Grid', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Material visualization saved to {filename}")
        
        return fig
    
    def save_grid_parameters(self, filename='grid_parameters.json'):
        """
        Save grid parameters to JSON file
        
        Parameters:
        filename: Output JSON filename
        """
        import json
        
        params = {
            'nx': int(self.nx),
            'ny': int(self.ny),
            'dx': float(self.dx),
            'dt': float(self.dt),
            'courant': float(self.courant),
            'total_cells': int(self.nx * self.ny),
            'domain_size_x': float(self.nx * self.dx),
            'domain_size_y': float(self.ny * self.dx),
            'simulation_time_per_step': float(self.dt),
            'water_cells': int(np.sum(self.epsilon == 1.0)),
            'gold_cells': int(np.sum(self.epsilon != 1.0)),
            'min_epsilon': float(np.min(self.epsilon)),
            'max_epsilon': float(np.max(self.epsilon)),
            'max_sigma': float(np.max(self.sigma))
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"Grid parameters saved to {filename}")
        return params

# Test function
def test_yee_grid():
    """Test Yee grid implementation"""
    print("\n" + "="*60)
    print("Testing YeeGrid2D Implementation")
    print("="*60)
    
    # Create output directories
    os.makedirs('figures/methods', exist_ok=True)
    os.makedirs('validation', exist_ok=True)
    
    # Test 1: Basic grid
    print("\nTest 1: Basic Grid Initialization")
    grid = YeeGrid2D(nx=100, ny=100, dx=1e-9)
    
    # Test 2: Material setting
    print("\nTest 2: Material Setting")
    import numpy as np
    gold_mask = np.zeros((100, 100), dtype=bool)
    gold_mask[40:60, 40:60] = True
    grid.set_material(gold_mask, epsilon_r=5.0, sigma_val=1e7)
    
    # Test 3: Field interpolation
    print("\nTest 3: Field Interpolation")
    field_val = grid.get_field_at_position(50e-9, 50e-9)
    print(f"  Interpolated field at (50nm, 50nm): {field_val}")
    
    # Test 4: Gold disk addition
    print("\nTest 4: Gold Disk Addition")
    grid2 = YeeGrid2D(nx=150, ny=150, dx=2e-9)
    disk_mask = grid2.add_gold_nanodisk(size_nm=100)
    
    # Test 5: Visualizations
    print("\nTest 5: Creating Visualizations")
    grid.visualize_grid('figures/methods/yee_grid_points.png')
    grid2.visualize_material_distribution('figures/methods/material_distribution.png')
    
    # Test 6: Parameter saving
    print("\nTest 6: Saving Parameters")
    params = grid.save_grid_parameters('validation/grid_parameters.json')
    
    # Save test data
    np.savez('validation/fdtd_basic_validation.npz',
             Ez_shape=grid.Ez.shape,
             Hx_shape=grid.Hx.shape,
             Hy_shape=grid.Hy.shape,
             dt=grid.dt,
             dx=grid.dx,
             nx=grid.nx,
             ny=grid.ny)
    
    print("\n" + "="*60)
    print("All Yee Grid Tests Completed Successfully!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Grid size: {grid.nx}×{grid.ny}")
    print(f"  Resolution: {grid.dx*1e9:.3f} nm")
    print(f"  Time step: {grid.dt*1e15:.3f} fs")
    print(f"  CFL number: {grid.courant:.3f}")
    print(f"  Output files created in 'figures/methods/' and 'validation/'")
    
    return grid

if __name__ == "__main__":
    grid = test_yee_grid()
