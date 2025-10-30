# run_experiments.py
# Convenience runner that imports verify_local_robustness.verify code (or calls the module main).
import subprocess
import os
import sys

# Simple wrapper: call the verify script which writes results CSV.
# You can further extend this to parallelize, filter models, etc.

SCRIPT = "verify_local_robustness.py"
if not os.path.exists(SCRIPT):
    print("ERROR: verify_local_robustness.py not found in current directory.")
    sys.exit(1)

print("Running experiments using", SCRIPT)
# Just run the script (it will write results to results/logs/results.csv)
ret = subprocess.run([sys.executable, SCRIPT], check=False)
print("verify_local_robustness.py exited with code", ret.returncode)
print("Results should be in results/logs/results.csv")
