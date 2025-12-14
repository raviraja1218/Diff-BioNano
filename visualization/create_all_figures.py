#!/usr/bin/env python3
"""
Master script to generate all Phase 4 figures
"""
import subprocess
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

def run_script(script_name):
    """Run a visualization script"""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run(['python', f'visualization/{script_name}'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            print(result.stdout[-500:])  # Last 500 chars of output
            return True
        else:
            print(f"❌ {script_name} failed")
            print("STDOUT:", result.stdout[-500:])
            print("STDERR:", result.stderr[-500:])
            return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Generate all figures"""
    scripts = [
        'design_evolution_sequence.py',    # Fig 2a
        'trajectory_overlay.py',           # Fig 2b
        'enhancement_comparison.py',       # Fig 3a  
        'spectral_comparison.py',          # Fig 3b
        'circular_dichroism.py',           # Fig 3c
        'hotspot_tracking.py',             # Fig 4a
        'frequency_matching.py',           # Fig 4b
        'mode_decomposition.py',           # Fig 4c
        'multi_molecule_test.py',          # Fig 5a
        'fabrication_tolerance.py',        # Fig 5b
        'application_scenarios.py',        # Fig 5c
        'optimization_convergence.py',     # Fig S5
        'radial_distribution.py',          # Fig S6
    ]
    
    print("="*60)
    print("DIFF-BIONANO: GENERATING ALL PAPER FIGURES")
    print("="*60)
    
    # First, create any missing scripts
    create_missing_scripts()
    
    # Run all scripts
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("FIGURE GENERATION SUMMARY")
    print("="*60)
    print(f"Successful: {success_count}/{len(scripts)}")
    
    if success_count == len(scripts):
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    else:
        print(f"⚠️  {len(scripts)-success_count} figures failed")
    
    # List generated files
    print("\nGenerated figures:")
    import os
    for root, dirs, files in os.walk("figures"):
        for file in files:
            if file.endswith('.png') or file.endswith('.gif'):
                print(f"  {os.path.join(root, file)}")

def create_missing_scripts():
    """Create any missing visualization scripts"""
    scripts_to_create = [
        'enhancement_comparison.py',
        'spectral_comparison.py',
        'circular_dichroism.py',
        'hotspot_tracking.py',
        'frequency_matching.py',
        'mode_decomposition.py',
        'multi_molecule_test.py',
        'fabrication_tolerance.py',
        'application_scenarios.py',
        'optimization_convergence.py',
        'radial_distribution.py',
    ]
    
    template = '''#!/usr/bin/env python3
"""
{description}
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Main function - to be implemented"""
    print("This script needs to be implemented")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Figure generation complete")
    else:
        print("❌ Figure generation failed")
        sys.exit(1)
'''
    
    for script in scripts_to_create:
        script_path = Path(f"visualization/{script}")
        if not script_path.exists():
            print(f"Creating template: {script}")
            with open(script_path, 'w') as f:
                f.write(template.format(description=f"Generate {script.replace('.py', '')}"))
            # Make executable
            script_path.chmod(0o755)

if __name__ == "__main__":
    main()
