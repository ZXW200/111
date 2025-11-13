"""
NTD Clinical Trials Analysis - Master Script
Group 16 | Lancaster University

This script executes the complete analysis pipeline in the correct order.
Useful for demonstration and reproducibility.
"""

import os
import sys
import time

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_script(script_name, description):
    """Execute a Python script and report results"""
    print(f"▶ Running: {script_name}")
    print(f"  Purpose: {description}")
    start_time = time.time()

    try:
        with open(script_name, 'r') as f:
            code = f.read()
        exec(code, {'__name__': '__main__'})
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f} seconds\n")
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        return False

def main():
    """Execute complete analysis pipeline"""

    print_header("NTD CLINICAL TRIALS ANALYSIS - FULL PIPELINE")
    print("Data Source: WHO ICTRP (1999-2023)")
    print("Research Questions: 5 RQs covering publication factors, networks,")
    print("                     pharma funding, special populations, and drug trends\n")

    # Check if raw data exists
    if not os.path.exists('ictrp_data.csv'):
        print("ERROR: ictrp_data.csv not found!")
        print("Please ensure the raw data file is in the current directory.")
        sys.exit(1)

    # Create output directories
    os.makedirs("CleanedData", exist_ok=True)
    os.makedirs("CleanedDataPlt", exist_ok=True)
    print("✓ Output directories created/verified\n")

    # Analysis pipeline
    scripts = [
        ("CleanData.py", "Data cleaning and preprocessing"),
        ("DataFit.py", "Logistic regression - publication factors (RQ 2.2.1)"),
        ("ExtractDrug.py", "Drug extraction - Chagas trends (RQ 2.2.5)"),
        ("Network.py", "Network analysis - country collaborations (RQ 2.2.2)"),
        ("visualization.py", "Visualizations - sponsor & regional analysis (RQ 2.2.3)")
    ]

    total_start = time.time()
    success_count = 0

    for i, (script, desc) in enumerate(scripts, 1):
        print_header(f"STEP {i}/5: {script}")
        if run_script(script, desc):
            success_count += 1
        else:
            print(f"WARNING: {script} failed. Continuing with remaining scripts...")

    # Summary
    total_elapsed = time.time() - total_start
    print_header("ANALYSIS COMPLETE")
    print(f"Successfully executed: {success_count}/{len(scripts)} scripts")
    print(f"Total execution time: {total_elapsed:.2f} seconds")

    # Check outputs
    print("\n--- Output Verification ---")
    csv_files = len([f for f in os.listdir('CleanedData') if f.endswith('.csv')])
    png_files = len([f for f in os.listdir('CleanedDataPlt') if f.endswith('.png')])
    print(f"CSV datasets generated: {csv_files}")
    print(f"Visualizations created: {png_files}")

    print("\n✓ Analysis pipeline complete!")
    print("  Check CleanedData/ for datasets")
    print("  Check CleanedDataPlt/ for visualizations\n")

if __name__ == "__main__":
    main()
