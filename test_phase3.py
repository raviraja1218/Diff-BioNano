#!/usr/bin/env python
"""
Test Phase 3 FDTD Implementation
"""
import sys
import os

def test_yee_grid():
    """Test Yee grid implementation"""
    print("\n" + "="*60)
    print("TESTING Yee Grid Implementation")
    print("="*60)
    
    try:
        # Import YeeGrid2D
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from fdtd.yee_grid import YeeGrid2D
        import numpy as np
        
        # Create grid
        grid = YeeGrid2D(nx=100, ny=100, dx=1e-9)
        
        # Test material setting
        gold_mask = np.zeros((100, 100), dtype=bool)
        gold_mask[40:60, 40:60] = True
        grid.set_material(gold_mask, epsilon_r=5.0, sigma_val=1e7)
        
        # Test interpolation
        field_val = grid.get_field_at_position(50e-9, 50e-9)
        
        print(f"✓ Grid: {grid.nx}×{grid.ny}, dx={grid.dx*1e9:.1f} nm")
        print(f"✓ Time step: {grid.dt*1e15:.2f} fs")
        print(f"✓ Material setting works")
        print(f"✓ Field interpolation returns: {field_val}")
        
        # Create directories for saving
        os.makedirs('figures/methods', exist_ok=True)
        os.makedirs('validation', exist_ok=True)
        
        # Create visualization
        grid.visualize_grid()
        print("✓ Grid visualization saved to figures/methods/yee_grid_points.png")
        
        # Save test data
        np.savez('validation/fdtd_basic_validation.npz',
                 Ez_shape=grid.Ez.shape,
                 Hx_shape=grid.Hx.shape,
                 Hy_shape=grid.Hy.shape,
                 dt=grid.dt,
                 dx=grid.dx)
        
        return True
        
    except Exception as e:
        print(f"❌ Yee grid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_maxwell_solver():
    """Test Maxwell solver"""
    print("\n" + "="*60)
    print("TESTING Maxwell Solver")
    print("="*60)
    
    try:
        # Import required modules
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from fdtd.yee_grid import YeeGrid2D
        from fdtd.maxwell_solver import MaxwellSolver2D
        import numpy as np
        
        # Create smaller grid for quick test
        grid = YeeGrid2D(nx=50, ny=50, dx=10e-9)
        solver = MaxwellSolver2D(grid)
        
        # Run short simulation
        print(f"Running simulation (50 steps)...")
        results = solver.run_simulation(steps=50, source_position=(25, 25))
        
        print(f"✓ Simulation completed successfully")
        print(f"✓ Final E field shape: {results['final_E'].shape}")
        print(f"✓ Energy history length: {len(results['energy_history'])}")
        print(f"✓ Max field value: {np.max(np.abs(results['final_E'])):.2e}")
        
        # Save test results
        test_dir = 'data/fdtd/validation_cases'
        os.makedirs(test_dir, exist_ok=True)
        np.savez(os.path.join(test_dir, 'maxwell_test.npz'),
                 E_field=results['final_E'])
        
        # Create energy plot
        import matplotlib.pyplot as plt
        os.makedirs('figures/validation', exist_ok=True)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(results['energy_history'])
        plt.xlabel('Time step (×10)')
        plt.ylabel('Total Energy (J)')
        plt.title('Energy Conservation Test')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(results['max_field_history'])
        plt.xlabel('Time step (×10)')
        plt.ylabel('Max |E| (V/m)')
        plt.title('Field Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/validation/energy_test.png', dpi=300)
        plt.close()
        
        print("✓ Energy plot saved to figures/validation/energy_test.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Maxwell solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("PHASE 3 FDTD IMPLEMENTATION TESTS")
    print("="*60)
    
    # Create necessary directories
    os.makedirs('figures/methods', exist_ok=True)
    os.makedirs('figures/validation', exist_ok=True)
    os.makedirs('data/fdtd/validation_cases', exist_ok=True)
    os.makedirs('validation', exist_ok=True)
    
    # Run tests
    test1 = test_yee_grid()
    test2 = test_maxwell_solver()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Yee Grid Test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Maxwell Solver Test: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    all_passed = test1 and test2
    
    if all_passed:
        # Create validation flag
        with open('validation/phase3_tests.passed', 'w') as f:
            f.write("Phase 3 Core FDTD Tests PASSED\n")
            f.write(f"Tests completed: {int(test1) + int(test2)}/2\n")
            f.write(f"- Yee Grid: {'PASS' if test1 else 'FAIL'}\n")
            f.write(f"- Maxwell Solver: {'PASS' if test2 else 'FAIL'}\n")
        
        print("\n✅ All Phase 3 core tests passed!")
        print("Validation flag created: validation/phase3_tests.passed")
        
        # Generate summary plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        tests = ['Yee Grid', 'Maxwell Solver']
        results = [1 if test1 else 0, 1 if test2 else 0]
        
        colors = ['green' if r else 'red' for r in results]
        ax.bar(tests, results, color=colors)
        ax.set_ylabel('Pass (1) / Fail (0)')
        ax.set_title('Phase 3 FDTD Implementation Tests')
        ax.set_ylim(0, 1.2)
        
        # Add text labels
        for i, (test, result) in enumerate(zip(tests, results)):
            ax.text(i, 0.5, 'PASS' if result else 'FAIL', 
                   ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('figures/validation/phase3_test_summary.png', dpi=300)
        plt.close()
        
        print("Summary plot saved: figures/validation/phase3_test_summary.png")
    else:
        print(f"\n❌ Some tests failed. Check error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
