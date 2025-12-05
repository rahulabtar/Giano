# Real-Time Piano Learning System - Hardware Integration

## Overview

This document describes the implementation of real-time bidirectional serial communication between the Raspberry Pi computer vision system and the glove/audio microcontrollers.

## Architecture

### Communication Flow

1. **Computer Vision Detection**
   - Camera captures frames at 30-60 FPS
   - MediaPipe detects hand landmarks
   - ArUco markers define piano boundaries
   - Fingers are tracked and mapped to piano keys

2. **State Tracking**
   - `FingerStateTracker` debounces finger positions (50ms)
   - Only sends updates when finger-key mapping changes
   - Maps MediaPipe finger IDs to motor IDs (0-4)

3. **Serial Communication**
   - Binary protocol: 3 bytes per message `[motor_id, midi_note, action]`
   - Non-blocking queues with threading
   - Auto-reconnection support
   - Simultaneous RX/TX for glove and audio board

4. **Glove Controller Response**
   - Receives finger-key updates
   - Provides haptic feedback based on actions
   - Sends acknowledgments back to Raspberry Pi

## Protocol Specification

### Glove Controller Protocol

**Message Format**: 3 bytes binary
```
[motor_id: u8, midi_note: u8, action: u8]
```

**Parameters**:
- `motor_id` (0-4): Thumb, Index, Middle, Ring, Pinky
- `midi_note` (0-127): MIDI note number of detected key
- `action` (0-5): Action code for haptic feedback

**Action Codes**:
- `0` (HOVER_GUIDE): Gentle vibration to guide finger to key
- `1` (PRESS_CORRECT): Strong positive feedback for correct key
- `2` (PRESS_INCORRECT): Warning vibration for incorrect key
- `3` (RELEASE): Stop vibration on key release
- `4` (NO_KEY): No key detected under finger
- `5` (ERROR): Error state

### Audio Board Protocol

**MIDI-Style Messages**: 3 bytes binary
```
[command: u8, note: u8, velocity: u8]
```

**Commands**:
- `0x90` (NOTE_ON): Start playing note
- `0x80` (NOTE_OFF): Stop playing note

## File Structure

### New Files Created

```
src/hardware/
  ├── __init__.py                  # Module exports
  ├── serial_manager.py            # Async serial I/O with threading
  └── protocols.py                 # Binary protocol definitions

src/computer_vision/
  └── finger_state_tracker.py     # Finger state management & debouncing

src/audio/
  └── audio_passthrough.py        # MIDI playback to audio board
```

### Modified Files

- `src/computer_vision/main.py`: Integrated serial communication and state tracking
- `config/default.json`: Added protocol constants and configuration

## Key Features

### 1. Non-Blocking Serial Communication

- **Send Thread**: Processes outgoing commands from queue
- **Receive Thread**: Reads incoming responses without blocking CV loop
- **Queue-based**: Messages queued when queue full (prevents blocking)

### 2. Debouncing and State Management

- **Debounce Time**: 50ms minimum between state changes
- **Change Detection**: Only sends updates when finger position changes
- **State Persistence**: Tracks last known key for each finger

### 3. Visual Feedback

- **FPS Display**: Top-right corner shows current frame rate
- **Connection Status**: Top-left shows "GA" (glove/audio connected) or "ga" (disconnected)

### 4. Error Handling

- **Graceful Degradation**: System continues if serial disconnects
- **Auto-Reconnect**: Attempts to reconnect devices automatically
- **Message Validation**: Checks message format before processing

## Usage

### Basic Usage

```python
python src/computer_vision/main.py
```

The system will:
1. Auto-detect serial devices (glove controller and audio board)
2. Start CV tracking loop
3. Send finger-key updates to glove controller
4. Display connection status on video feed

### Advanced Configuration

Edit `config/default.json` to configure:
- Serial ports (auto-detect or manual)
- Baud rates
- Debounce timing
- Action codes

## Performance Characteristics

- **CV Loop**: 30-60 FPS (dependent on camera and processing)
- **Serial Latency**: <10ms with queued messages
- **Debounce Period**: 50ms (prevents excessive messaging)
- **Protocol Efficiency**: 3 bytes per message vs ~20 bytes text protocol

## Microcontroller Requirements

### Glove Controller (Teensy)

The microcontroller must implement:

```cpp
void loop() {
    if (Serial.available() >= 3) {
        uint8_t motor_id = Serial.read();
        uint8_t midi_note = Serial.read();
        uint8_t action = Serial.read();
        
        // Process haptic feedback
        handle_haptic_command(motor_id, midi_note, action);
        
        // Send acknowledgment (optional)
        Serial.write(motor_id);
        Serial.write(midi_note);
        Serial.write(action);
    }
}
```

### Audio Board (Teensy)

The microcontroller must implement:

```cpp
void loop() {
    if (Serial.available() >= 3) {
        uint8_t command = Serial.read();
        uint8_t note = Serial.read();
        uint8_t velocity = Serial.read();
        
        if (command == 0x90) {  // NOTE_ON
            play_note(note, velocity);
        } else if (command == 0x80) {  // NOTE_OFF
            stop_note(note);
        }
    }
}
```

## Testing

### Without Hardware

The system will work without connected devices:
- CV tracking continues normally
- Connection status shows "ga" (disconnected)
- No serial errors (gracefully handles missing devices)

### With Hardware

1. Connect glove controller to USB
2. Connect audio board to USB
3. Run `main.py`
4. Place hands over piano with ArUco markers visible
5. Observe haptic feedback as fingers move over keys

## Troubleshooting

### Serial Connection Issues

- Check USB connections
- Verify baud rate (115200)
- Use `ls /dev/tty*` (Linux) or check Device Manager (Windows)
- Try manual port specification in code

### No Finger Detection

- Ensure ArUco markers are visible and correctly positioned
- Check camera calibration
- Verify MediaPipe model is loaded

### Delayed Haptic Feedback

- Increase debounce time (reduce sensitivity)
- Check serial queue status in logs
- Verify microcontroller processing speed

## Future Enhancements

1. **Audio Passthrough Integration**: Real-time MIDI playback synchronization
2. **Multi-Hand Support**: Track left and right hands separately
3. **State Machine**: Advanced finger state transitions (press, release, errors)
4. **Machine Learning**: Adaptive haptic feedback based on performance
5. **Visual Piano**: Draw detected keys on video feed

## References

- MediaPipe Hand Landmarks: https://google.github.io/mediapipe/solutions/hands.html
- ArUco Markers: https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
- Serial Communication: https://pyserial.readthedocs.io/
- MIDI Protocol: https://www.midi.org/specifications-old/item/midi-message-format

