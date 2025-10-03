# Giano System Setup Guide

## Prerequisites

- Python 3.8 or higher
- PlatformIO for hardware development
- Camera (USB webcam or integrated)
- Teensy 4.0 boards (2x)
- Required hardware components (see hardware documentation)

## Installation Steps

### 1. Clone Repository
```bash
git clone [your-repo-url]
cd Giano
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python -m venv giano_env

# Activate virtual environment
# Windows:
giano_env\Scripts\activate
# Linux/Mac:
source giano_env/bin/activate

# Install base dependencies
pip install -r requirements/base.txt
```

### 3. Module-Specific Setup

#### Computer Vision Module
```bash
pip install -r requirements/cv.txt

# Download MediaPipe hand model (if not included)
# Place hand_landmarker.task in assets/models/
```

#### Audio Module
```bash
pip install -r requirements/audio.txt
```

#### Development Tools
```bash
pip install -r requirements/dev.txt
```

### 4. Hardware Setup

#### Teensy Boards
1. Install PlatformIO extension in VS Code
2. Open `hardware/teensy/glove/` project
3. Build and upload to glove controller Teensy
4. Open `hardware/teensy/audio/` project  
5. Build and upload to audio board Teensy

#### Camera Calibration
```bash
python scripts/calibrate_camera.py
```

### 5. Configuration

Copy example configs and modify for your setup:
```bash
cp config/default.json.example config/default.json
cp config/camera.json.example config/camera.json
cp config/audio.json.example config/audio.json
```

Edit configuration files with your specific parameters.

### 6. Verification

Run system tests:
```bash
python -m pytest tests/
```

Run examples to verify setup:
```bash
python examples/basic_hand_tracking.py
python examples/midi_playback.py
```

## Troubleshooting

### Common Issues

1. **MediaPipe model not found**
   - Ensure `hand_landmarker.task` is in `assets/models/`
   - Download from MediaPipe documentation if missing

2. **Camera not detected** 
   - Check camera permissions
   - Try different camera IDs in config
   - Verify camera works with other applications

3. **Teensy upload fails**
   - Check USB connection
   - Verify correct board selected in PlatformIO
   - Try different USB cable/port

4. **Import errors**
   - Ensure virtual environment is activated
   - Verify all requirements installed correctly
   - Check Python path includes project root

## Development Workflow

1. Make changes to source code
2. Run tests: `python -m pytest tests/`
3. Format code: `black src/`
4. Check linting: `flake8 src/`
5. Type check: `mypy src/`