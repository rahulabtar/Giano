#!/usr/bin/env python3
"""
Project management utilities for Giano system.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the test suite."""
    print("Running Giano test suite...")
    test_dir = Path(__file__).parent.parent / "tests"
    if test_dir.exists():
        cmd = f"{sys.executable} -m pytest {test_dir} -v"
        subprocess.run(cmd, shell=True)
    else:
        print("No tests directory found. Create tests/ directory and add test files.")

def format_code():
    """Format code using black."""
    print("Formatting code with black...")
    src_dir = Path(__file__).parent.parent / "src"
    cmd = f"{sys.executable} -m black {src_dir}"
    subprocess.run(cmd, shell=True)

def lint_code():
    """Lint code using flake8."""
    print("Linting code with flake8...")
    src_dir = Path(__file__).parent.parent / "src"
    cmd = f"{sys.executable} -m flake8 {src_dir}"
    subprocess.run(cmd, shell=True)

def type_check():
    """Type check code using mypy."""
    print("Type checking with mypy...")
    src_dir = Path(__file__).parent.parent / "src"
    cmd = f"{sys.executable} -m mypy {src_dir}"
    subprocess.run(cmd, shell=True)

def clean_project():
    """Clean generated files."""
    print("Cleaning project...")
    project_root = Path(__file__).parent.parent
    
    # Remove Python cache files
    for cache_dir in project_root.rglob("__pycache__"):
        print(f"Removing {cache_dir}")
        import shutil
        shutil.rmtree(cache_dir)
    
    # Remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        print(f"Removing {pyc_file}")
        pyc_file.unlink()

def check_dependencies():
    """Check if all dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        "numpy", "opencv-python", "mediapipe", "music21", 
        "matplotlib", "pillow", "sounddevice"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run 'python scripts/install.py' to install dependencies")
    else:
        print("\n✓ All core dependencies are installed!")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Giano Project Management")
    parser.add_argument("command", choices=[
        "test", "format", "lint", "typecheck", "clean", "deps", "install"
    ], help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "test":
        run_tests()
    elif args.command == "format":
        format_code()
    elif args.command == "lint":
        lint_code()
    elif args.command == "typecheck":
        type_check()
    elif args.command == "clean":
        clean_project()
    elif args.command == "deps":
        check_dependencies()
    elif args.command == "install":
        from . import install
        install.main()

if __name__ == "__main__":
    main()