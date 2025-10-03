# Giano - Guided Piano Gloves System

A comprehensive system for piano learning using computer vision, haptic feedback gloves, and audio synthesis. Created by  

## Project Structure

```
Giano/
├── src/                     # Main source code
│   ├── computer_vision/     # Hand tracking and ArUco markers
│   ├── audio/              # MIDI processing and synthesis  
│   ├── integration/        # System coordination
│   └── core/               # Shared utilities
├── hardware/               # Microcontroller firmware
│   ├── teensy/            # Teensy 4.0 projects
│   └── arduino/           # Arduino projects
├── assets/                # Static files and models
├── config/                # Configuration files
├── examples/              # Usage examples
├── docs/                  # Documentation
└── requirements/          # Dependencies by module
```

## Quick Start

1. **Install Dependencies**
   ```bash
   # For computer vision module
   pip install -r requirements/cv.txt
   
   # For audio module  
   pip install -r requirements/audio.txt
   
   # For development
   pip install -r requirements/dev.txt
   ```

2. **Hardware Setup**
   - Flash Teensy boards with firmware from `hardware/teensy/`
   - Calibrate camera using `src/computer_vision/camera_calibration.py`

3. **Run Examples**
   ```bash
   # Basic hand tracking
   python examples/basic_hand_tracking.py
   
   # MIDI file processing
   python examples/midi_playback.py
   

## System Components

### Computer Vision (`src/computer_vision/`)
- Hand landmark detection using MediaPipe
- ArUco marker tracking for piano key mapping
- Camera calibration and pose estimation

### Audio Processing (`src/audio/`)
- MIDI file parsing and note extraction
- Fingering and hand placement analysis

### Hardware Integration (`hardware/`)
- **Glove Controller**: Haptic feedback motors controlled via Teensy
- **Audio Board**: Real-time audio synthesis and MIDI processing
- **Flex Sensors**: Finger bend detection


## Development

See `docs/setup.md` for detailed development setup instructions.
See `docs/architecture.md` for system architecture details.

