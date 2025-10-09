#!/usr/bin/env python3
"""
Migration script to reorganize existing Giano project structure.
Run this script to move files from the old structure to the new organized structure.
"""

import shutil
import os
from pathlib import Path

def migrate_project():
    """Migrate existing files to new structure."""
    
    print("Starting Giano project migration...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create backup
    backup_dir = project_root / "backup_old_structure"
    if not backup_dir.exists():
        print("Creating backup of old structure...")
        backup_dir.mkdir()
    
    # Migration mappings (old_path -> new_path)
    migrations = [
        # Computer Vision files
        ("giano_raspi/giano_cv.py", "src/computer_vision/main.py"),
        ("giano_raspi/calibrate_camera.py", "src/computer_vision/calibration.py"),
        ("giano_raspi/camera_calibration.npz", "assets/calibration/camera_calibration.npz"),
        ("giano_raspi/camera_calibration_live.npz", "assets/calibration/camera_calibration_live.npz"),
        ("giano_raspi/calibration_images", "assets/calibration_images"),
        
        # ArUco utilities
        ("giano_utils/giano_aruco.py", "src/computer_vision/aruco_system.py"),
        ("giano_utils/aruco_pose_tracker.py", "src/computer_vision/aruco_tracker.py"),
        ("giano_utils/aruco_polygon_detector.py", "src/computer_vision/aruco_detector.py"),
        ("giano_utils/aruco_add_to_sheet.py", "src/computer_vision/aruco_sheet.py"),
        ("giano_utils/aruco_marker_generator.py", "scripts/generate_aruco_markers.py"),
        ("giano_utils/aruco_output", "assets/aruco_markers"),
        ("giano_utils/aruco_input", "assets/aruco_input"),
        
        # Audio processing
        ("MIDI_Song_Decoder/MusicDecoder.py", "src/audio/midi_decoder.py"),
        ("MIDI_Song_Decoder/example.py", "examples/midi_playback.py"),
        ("MIDI_Song_Decoder/MXLFiles", "assets/midi_files"),
        
        # Hardware projects  
        ("giano_teensy/glove", "hardware/teensy/glove"),
        ("giano_teensy/audio", "hardware/teensy/audio"),
        ("Flex_Resistor_Testing", "hardware/arduino/flex_sensor"),
        
        # Other assets
        ("giano_raspi/z_axis_stability.txt", "assets/calibration/z_axis_stability.txt"),
    ]
    
    # Perform migrations
    for old_path_str, new_path_str in migrations:
        old_path = project_root / old_path_str
        new_path = project_root / new_path_str
        
        if old_path.exists():
            print(f"Moving {old_path_str} -> {new_path_str}")
            
            # Create parent directory if needed
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy to backup first
            backup_path = backup_dir / old_path_str
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            if old_path.is_dir():
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(old_path, backup_path)
                
                if new_path.exists():
                    shutil.rmtree(new_path)
                shutil.move(old_path, new_path)
            else:
                shutil.copy2(old_path, backup_path)
                shutil.move(old_path, new_path)
        else:
            print(f"Warning: {old_path_str} not found, skipping...")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/core/__init__.py", 
        "src/computer_vision/__init__.py",
        "src/audio/__init__.py",
        "src/integration/__init__.py",
    ]
    
    for init_file in init_files:
        init_path = project_root / init_file
        if not init_path.exists():
            init_path.touch()
            print(f"Created {init_file}")
    
    print("\nMigration complete!")
    print(f"Backup of old structure saved in: {backup_dir}")
    print("\nNext steps:")
    print("1. Review migrated files for any needed adjustments")
    print("2. Update import statements in Python files") 
    print("3. Test the new structure with examples/")
    print("4. Remove old empty directories when satisfied")

if __name__ == "__main__":
    migrate_project()