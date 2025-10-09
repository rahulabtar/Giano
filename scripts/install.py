#!/usr/bin/env python3
"""
Installation and setup script for Giano system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, check=True, shell=True, 
                              capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_requirements(req_file, description):
    """Install requirements from a file."""
    req_path = Path(__file__).parent.parent / "requirements" / req_file
    if req_path.exists():
        command = f"{sys.executable} -m pip install -r {req_path}"
        return run_command(command, f"Installing {description}")
    else:
        print(f"Warning: Requirements file {req_file} not found")
        return False

def main():
    """Main installation process."""
    print("Giano System Installation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    success_count = 0
    total_count = 0
    
    # Base requirements
    total_count += 1
    if install_requirements("base.txt", "base dependencies"):
        success_count += 1
    
    # Computer vision requirements
    total_count += 1
    if install_requirements("cv.txt", "computer vision dependencies"):
        success_count += 1
    
    # Audio requirements  
    total_count += 1
    if install_requirements("audio.txt", "audio processing dependencies"):
        success_count += 1
    
    # ArUco requirements
    total_count += 1
    if install_requirements("aruco.txt", "ArUco marker dependencies"):
        success_count += 1
    
    # Development requirements (optional)
    print("\nOptional development tools:")
    install_requirements("dev.txt", "development tools (optional)")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Installation Summary: {success_count}/{total_count} modules installed successfully")
    
    if success_count == total_count:
        print("✓ All core modules installed successfully!")
        print("\nNext steps:")
        print("1. Review configuration files in config/")
        print("2. Run camera calibration if needed")
        print("3. Test with examples in examples/")
    else:
        print("⚠ Some modules failed to install. Check error messages above.")
        print("You may need to install some dependencies manually.")
    
    print("\nFor detailed setup instructions, see docs/setup.md")

if __name__ == "__main__":
    main()