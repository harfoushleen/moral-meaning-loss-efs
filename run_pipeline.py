#!/usr/bin/env python
"""
Master pipeline orchestrator.
Run all configured steps in sequence with error-checking and logging.

Usage:
    python run_pipeline.py                  # Run all steps
    python run_pipeline.py --from 2         # Start from step 2
    python run_pipeline.py --only 3         # Run only step 3
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Pipeline steps in order
STEPS = [
    ("00_download_books.py", "Download books from Project Gutenberg"),
    ("01_extract_passages.py", "Extract ~50 passages per book"),
    ("02_generate_interpretations.py", "Generate LLM interpretations (~900 calls)"),
    ("03_extract_features.py", "Extract 4 EFS feature dimensions"),
    ("04_compute_efs.py", "Compute Epistemic Fidelity Scores"),
    ("05_analyze_results.py", "Generate figures and statistics"),
    ("06_validate_metrics.py", "Run ablation, baseline, sensitivity, and extreme-case analyses"),
]

SRC_DIR = Path(__file__).parent / "src"


def run_step(step_num: int, script_name: str, description: str) -> bool:
    """Run a single pipeline step. Return True if successful."""
    step_path = SRC_DIR / script_name
    
    if not step_path.exists():
        print(f"\n❌ FAILED: {step_path} not found")
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[{timestamp}] Step {step_num}: {description}")
    print(f"Running: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(step_path)],
            check=True,
            cwd=str(SRC_DIR.parent)
        )
        print(f"\n✅ Step {step_num} PASSED\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_num} FAILED with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Step {step_num} FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the moral meaning loss pipeline."
    )
    parser.add_argument(
        "--from",
        type=int,
        dest="from_step",
        default=0,
        help="Start from step N (default: 0)"
    )
    parser.add_argument(
        "--only",
        type=int,
        dest="only_step",
        default=None,
        help="Run only step N"
    )
    args = parser.parse_args()
    
    start = args.from_step
    end = args.only_step if args.only_step is not None else len(STEPS) - 1
    
    # Validate range
    if not (0 <= start <= len(STEPS) - 1):
        print(f"❌ Invalid start step. Must be 0–{len(STEPS)-1}")
        sys.exit(1)
    if args.only_step is not None and not (0 <= args.only_step <= len(STEPS) - 1):
        print(f"❌ Invalid only step. Must be 0–{len(STEPS)-1}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Moral Meaning Loss in LLMs — Pipeline Orchestrator")
    print(f"{'='*70}")
    print(f"\nSteps to run:")
    for i in range(start, end + 1):
        script, desc = STEPS[i]
        print(f"  {i}: {desc}")
    
    input("\nPress Enter to begin...")
    
    passed = 0
    failed = 0
    failed_steps = []
    
    for i in range(start, end + 1):
        script, desc = STEPS[i]
        if run_step(i, script, desc):
            passed += 1
        else:
            failed += 1
            failed_steps.append((i, script))
            print(f"\n⚠️  Pipeline halted at step {i}.")
            print(f"   Fix the error and re-run with: python run_pipeline.py --from {i}")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Pipeline Summary")
    print(f"{'='*70}")
    print(f"✅ Passed: {passed} steps")
    print(f"❌ Failed: {failed} steps")
    
    if failed_steps:
        print(f"\nFailed steps:")
        for step_num, script in failed_steps:
            print(f"  {step_num}: {script}")
        print(f"\nTo resume from step {failed_steps[0][0]}, run:")
        print(f"  python run_pipeline.py --from {failed_steps[0][0]}")
        sys.exit(1)
    else:
        print(f"\n🎉 All steps completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Check outputs in data/results/ and outputs/figures/")
        print(f"  2. (Optional) Annotate passages: see ANNOTATION_GUIDE.md")
        print(f"  3. Generate paper figures and tables")
        sys.exit(0)


if __name__ == "__main__":
    main()
