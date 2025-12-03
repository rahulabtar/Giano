"""
Serial communication manager for hardware devices.

Provides async I/O with threading, device discovery, connection management,
and non-blocking send/receive queues.
"""

import serial
import sys
import glob
import threading
import queue
import time
import logging
from typing import Optional, List, Dict, Callable, Literal, Union
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


from .protocols import GloveProtocolFreeplayMode, GloveProtocolLearningMode, AudioProtocol, PlayingMode, Hand, SensorValue, SensorNumberLeft

from src.core.constants import SERIAL_BAUD_RATE, LEFT_PORT, RIGHT_PORT


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SWITCH PORTS AND DURING CONNECTION RETURN THE HAND
# LEFT SETUP WILL READ THE PLAYING MODE FROM THE GLOVE CONTROLLER

class BaseSerialManager:
    """
    Base class for serial communication managers.
    
    Provides common functionality for port discovery, connection management,
    and thread lifecycle.
    """
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200):
        """
        Initialize base serial manager.
        
        Args:
            port: Serial port (None for auto-detect)
            baud_rate: Baud rate for serial communication
        """
        self.port = port
        self.baud_rate = baud_rate
        self.conn: Optional[serial.Serial] = None
        self._running = False
    
    def _list_serial_ports(self) -> List[str]:
        """List all available serial ports."""
        if sys.platform.startswith('win'):
            ports = [f'COM{i}' for i in range(1, 256)]
        elif sys.platform.startswith('linux'):
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            ports = []
        
        available = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                available.append(port)
            except:
                pass
        
        return available
    
    def _connect(self, port: str) -> tuple[bool, Optional[Hand]]:
        """
        Connect to device on specified port.
        
        Returns:
            Tuple of (success: bool, detected_hand: Optional[Hand])
            - success: True if connection successful
            - detected_hand: The hand detected during connection, or None if failed/not a glove
        """
        try:
            self.conn = serial.Serial(port, self.baud_rate, timeout=0.1)
            self.port = port
            hand_bytes = self.conn.read(1)
            if not hand_bytes:
                logger.warning("No hand bytes received from device")
                return False, None
            hand_value = hand_bytes[0]
            if hand_value == Hand.LEFT.value:
                return True, Hand.LEFT
            elif hand_value == Hand.RIGHT.value:
                return True, Hand.RIGHT
            else:
                logger.warning(f"Invalid hand value: {hand_value}")
                return False, None
        except Exception as e:
            logger.error(f"Failed to connect on {port}: {e}")
            return False, None
    
    def disconnect(self):
        """Disconnect from device."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def is_connected(self) -> bool:
        """Check if connected to device."""
        return self.conn is not None and self.conn.is_open
    
    def _test_port(self, port: str) -> bool:
        """
        Test if a port is the correct device.
        
        Subclasses should override this method to implement device-specific
        identification logic. Default implementation just checks if the port
        can be opened.
        
        Args:
            port: Serial port to test
            
        Returns:
            True if port matches the device, False otherwise
        """
        try:
            s = serial.Serial(port, self.baud_rate, timeout=0.1)
            s.close()
            return True
        except:
            return False
    
    def _auto_connect(self) -> tuple[bool, Optional[Hand]]:
        """
        Auto-detect and connect to device by testing available ports.
        
        Iterates through available serial ports and tests each one using
        _test_port(). Connects to the first port that passes the test.
        
        Returns:
            Tuple of (success: bool, detected_hand: Optional[Hand])
            - success: True if connection successful, False otherwise
            - detected_hand: The hand detected during connection, or None if failed
        """
        available_ports = self._list_serial_ports()
        logger.info(f"Available serial ports: {available_ports}")
        
        for port in available_ports:
            if self._test_port(port):
                success, detected_hand = self._connect(port)
                if success:
                    return success, detected_hand
        
        logger.warning("Could not auto-detect device")
        return False, None
    
    def stop(self):
        """Stop all communication threads."""
        self._running = False
        self.disconnect()

""" 
Glove connection / hand-detection process
 1. BaseSerialManager._connect(port) opens the serial port and reads 1 byte to
    detect the hand: Hand.LEFT or Hand.RIGHT.
 2. BaseSerialManager._auto_connect() scans ports, uses _test_port(), then
    calls _connect() and returns (success, detected_hand).
 3. LeftGloveSerialManager.connect():
    - Calls _connect() / _auto_connect().
    - If detected_hand != self.hand:
        * Logs a warning.
        * Uses create_for_hand(detected_hand, port, baud) to build the correct
          manager (LeftGloveSerialManager or RightGloveSerialManager).
        * Transfers the open serial connection to that manager (no reconnect),
          starts it, and returns it.
    - If detected_hand == self.hand:
        * Updates self.hand, calls start(), and returns self. 
"""


# MAX OF 8? TRIES OF CONNECTING
class LeftGloveSerialManager(BaseSerialManager):
    """
    Manages serial communication with left glove controller for haptic feedback.
    """
    
    def __init__(self, 
                 port: Optional[str] = None, 
                 baud_rate: int = SERIAL_BAUD_RATE, 
                ):
        """
        Initialize glove serial manager.
        
        Args:
            port: Serial port for glove controller (None for auto-detect)
            baud_rate: Baud rate for serial communication
        """
        super().__init__(port, baud_rate)
        
        # Queues for non-blocking I/O
        self.send_queue = queue.Queue(maxsize=100)
        self.recv_queue = queue.Queue(maxsize=100)
        
        # Threads
        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        
        # Callback for incoming messages
        self.callback: Optional[Callable] = None
        
        self.hand = Hand.LEFT
        self._play_mode: Optional[PlayingMode] = None
    
    @classmethod
    def create_for_hand(cls, hand: Hand, port: Optional[str] = None, baud_rate: int = SERIAL_BAUD_RATE):
        """
        Factory method to create the correct glove manager for a detected hand.
        
        Args:
            hand: The detected hand (Hand.LEFT or Hand.RIGHT)
            port: Serial port (None for auto-detect)
            baud_rate: Baud rate for serial communication
            
        Returns:
            LeftGloveSerialManager or RightGloveSerialManager instance
        """
        if hand == Hand.LEFT:
            return LeftGloveSerialManager(port=port, baud_rate=baud_rate)
        elif hand == Hand.RIGHT:
            return RightGloveSerialManager(port=port, baud_rate=baud_rate)
        else:
            raise ValueError(f"Invalid hand: {hand}")
    
    def connect(self) -> tuple[bool, Optional[Hand], Optional[Union['LeftGloveSerialManager', 'RightGloveSerialManager']]]:
        """
        Connect to glove controller. If the wrong hand is detected, returns
        a new manager instance for the correct hand with the connection transferred.
        
        MUST BE CALLED BEFORE start()
        
        Returns:
            Tuple of (success: bool, detected_hand: Optional[Hand], correct_manager: Optional[Union[LeftGloveSerialManager, RightGloveSerialManager]])
            - success: True if connection successful
            - detected_hand: The hand detected during connection
            - correct_manager: If wrong hand detected, returns the correct manager type (LeftGloveSerialManager or RightGloveSerialManager) with connection transferred. Otherwise None.
        """
        if self.port:
            success, detected_hand = self._connect(self.port)
        else:
            success, detected_hand = self._auto_connect()
        
        if not success:
            return False, None, None
        
        # Check if we connected to the wrong hand
        if detected_hand != self.hand:
            logger.warning(
                f"Connected to {detected_hand}-hand glove but this is a {self.hand}-hand manager. "
                f"Switching to correct manager type."
            )
            # Transfer the connection to the correct manager type
            correct_manager = self.create_for_hand(detected_hand, port=self.port, baud_rate=self.baud_rate)
            # Transfer the connection (don't disconnect, just transfer)
            correct_manager.conn = self.conn
            correct_manager.port = self.port
            correct_manager.hand = detected_hand
            # Clear our connection so we don't close it
            self.conn = None
            # Start the correct manager
            correct_manager.start()
            return True, detected_hand, correct_manager
        
        # Correct hand detected, update self.hand and start
        self.hand = detected_hand
        self.start()
        return True, detected_hand, None
    
    def _test_port(self, port: str) -> bool:
        """
        Test if port is glove controller.
        
        Note: This consumes the hand identifier byte from the serial stream.
        If the glove only sends this byte once on startup, repeated calls to
        _auto_connect() may fail to detect the correct port.
        """
        try:
            s = serial.Serial(port, self.baud_rate, timeout=0.1)
            # Read hand identifier byte (consumes it from the stream)
            glove_hand_bytes = s.read(1)
            s.close()
            
            if not glove_hand_bytes:
                return False
            
            # Convert bytes to int for comparison with IntEnum
            glove_hand_value = glove_hand_bytes[0]
            
            print(f"Glove hand value: {glove_hand_value}")
            print(f"Expected hand: {self.hand}")

            # Check if this port matches the expected hand
            if glove_hand_value == Hand.LEFT.value and self.hand == Hand.LEFT:
                logger.info(f"Found {self.hand}-hand glove controller on {port}")
                return True
            elif glove_hand_value == Hand.RIGHT.value and self.hand == Hand.RIGHT:
                logger.info(f"Found correct {self.hand}-hand glove controller on {port}")
                return True
            else:
                # Update hand if we found a different one
                detected_hand = Hand(glove_hand_value)
                logger.warning(f"Found {detected_hand}-hand glove controller on {port}, expected {self.hand}")
                self.hand = detected_hand
                return False
        except Exception as e:
            logger.debug(f"Error testing port {port}: {e}")
            return False
    
    def start(self):
        """
        Start all communication threads.

        This will correspond to setup on the left glove controller.
        
        Note: This consumes the mode byte from the serial stream.
        If the glove only sends this byte once on startup, ensure it's
        available when start() is called.
        """
        if self._running or not self.conn:
            return
        
        if self.conn.in_waiting > 0:
            # Read the first byte in stream to get the mode info (consumes it)
            mode_bytes = self.conn.read(1)
            
            if not mode_bytes:
                logger.warning("No mode byte received from glove controller")
                return
            
            # Convert bytes to int for comparison
            mode_value = mode_bytes[0]
            
            print(f"Mode: {mode_value} from {self.hand}-hand glove controller")
            
            # PlayingMode is a regular Enum, so compare with .value
            if mode_value == PlayingMode.LEARNING_MODE.value:
                self._play_mode = PlayingMode.LEARNING_MODE
            elif mode_value == PlayingMode.FREEPLAY_MODE.value:
                self._play_mode = PlayingMode.FREEPLAY_MODE
            else:
                raise ValueError(f"Invalid play mode: {mode_value}")

        self._running = True
        
        self._send_thread = threading.Thread(
            target=self._send_worker, daemon=True)
        self._recv_thread = threading.Thread(
            target=self._recv_worker, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()
        
        logger.info("Glove serial manager started")
    
    def stop(self):
        """Stop all communication threads."""
        super().stop()
        
        if self._send_thread:
            self._send_thread.join(timeout=1)
        if self._recv_thread:
            self._recv_thread.join(timeout=1)
        
        logger.info("Glove serial manager stopped")
    
    def handle_line_rx(self, line: bytes):
        """
        line = bytes message like: b'L 0 3' in the format of [hand, sensorValue, sensorNumber]
        Returns:
            instruction: FreeplayModeGloveInstructionSet object
        """
        print(f"Received from {self.hand}-Teensy:", line)

        if self._play_mode == PlayingMode.FREEPLAY_MODE:
            instruction = GloveProtocolFreeplayMode.unpack(line)
            if instruction.hand != self.hand:
                logger.warning(f"Found incorrect {instruction.hand}-hand glove controller on {self.port}, switching hand info...")
                self.hand = instruction.hand
                return None        

            if instruction.sensorValue == SensorValue.Pressed: # we can use this to then map to cv calls to assign a note and play it
                print(f"[{instruction.hand}] Sensor {instruction.sensorNumber} PRESSED") #instead of just printing out
                

            elif instruction.sensorValue == SensorValue.Released: # this we can use to map to a note and turn it off
                print(f"[{instruction.hand}] Sensor {instruction.sensorNumber} RELEASED") #instead of just printing it otu

        elif self._play_mode == PlayingMode.LEARNING_MODE:
            instruction = GloveProtocolLearningMode.unpack(line)
            # TODO: Implement learning mode handling
            return instruction
        else:
            raise ValueError(f"Invalid play mode: {self._play_mode}")

        
            # noteOff(hand, sensorIdx)

        return instruction

    def send_command(self, motor_id: int, midi_note: int, action: int):
        """
        Send command to glove controller (non-blocking).
        
        Args:
            motor_id: Motor ID (0-4)
            midi_note: MIDI note number
            action: Action code
        """
        if self._play_mode == PlayingMode.FREEPLAY_MODE:
            # TODO: this is wrong for freeplay mode
            message = GloveProtocolFreeplayMode.pack(motor_id, midi_note, action)
        elif self._play_mode == PlayingMode.LEARNING_MODE:
            message = GloveProtocolLearningMode.pack(motor_id, midi_note, action)
        else:
            raise ValueError(f"Invalid play mode: {self._play_mode}")
        try:
            self.send_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Glove send queue full, dropping message")
    
    def get_responses(self) -> List[bytes]:
        """Get all pending glove responses."""
        responses = []
        while not self.recv_queue.empty():
            try:
                responses.append(self.recv_queue.get_nowait())
            except queue.Empty:
                break
        return responses
    
    def set_callback(self, callback: Callable):
        """Set callback for incoming glove messages."""
        self.callback = callback
    
    def _send_worker(self):
        """Worker thread for sending to glove controller."""
        while self._running:
            try:
                message = self.send_queue.get(timeout=0.1)
                if self.conn:
                    self.conn.write(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending glove message: {e}")
    
    def _recv_worker(self):
        """Worker thread for receiving from glove controller."""
        while self._running:
            try:
                if self.conn and self.conn.in_waiting > 3: # 3 bytes is the size of the message
                    line = self.conn.read(3)
                    self.recv_queue.put_nowait(line)
                    # TODO: See where handling lines should go
                    # self.handle_line_rx(line)

                    # Also call callback if set
                    if self.callback:
                        self.callback(line)
            except queue.Full:
                logger.warning("Glove recv queue full, dropping data")
            except Exception as e:
                logger.error(f"Error receiving glove message: {e}")
                time.sleep(0.1)


class RightGloveSerialManager(LeftGloveSerialManager):
    """
    Manages serial communication with right glove controller for haptic feedback.
    
    Inherits from LeftGloveSerialManager and only changes the default hand.
    """
    
    def __init__(self, 
                 port: Optional[str] = None, 
                 baud_rate: int = SERIAL_BAUD_RATE, 
                ):
        """
        Initialize right glove serial manager.
        
        Args:
            port: Serial port for glove controller (None for auto-detect)
            baud_rate: Baud rate for serial communication
        """
        super().__init__(port, baud_rate)
        self.hand = Hand.RIGHT
    
    def start(self):
        """
        Start all communication threads.

        This will correspond to setup on the right glove controller.
        
        Note: This consumes the mode byte from the serial stream.
        If the glove only sends this byte once on startup, ensure it's
        available when start() is called.
        """
        super().start()  # Call parent implementation



# Legacy functions for backward compatibility

def read_from_teensy(port, source_hand):
    """Run in its own thread to read from a single teensy"""
    ser = serial.Serial(port, SERIAL_BAUD_RATE)
    print(f"Listening on {port} for {source_hand}-hand Teensy...")

    while True:
        try:
            line = ser.readline().decode().strip()
            # TODO: Implement handle_line function or remove this legacy code
            # handle_line(line)
        except Exception as e:
            print(f"Error on {source_hand}-hand port: {e}")



class AudioProtocol:
    
    def __init__(self, output = 'Teensy MIDI/Audio'):
        self.out = mido.open_output(output)

    def note_on(self, note: int, velocity: int = 100):
        #if (note < 60): return #values less than 60 are reserved for voice commands
        if note > 127:
            raise ValueError(f"Invalid MIDI note value: {note}. MIDI notes must be 0-127.")
        velocity = 127 if velocity > 127 else velocity
        velocity = 0 if velocity < 0 else velocity
        self.out.send(mido.Message('note_on', note=note, velocity=velocity, channel=0))
        #print("I turned on note " + str(note) + " with velocity " +  str(velocity))
    
    def note_off(self, note: int, velocity: int = 0):
        #if (note < 60): return #values less than 60 are reserved for voice commands
        if note > 127:
            raise ValueError(f"Invalid MIDI note value: {note}. MIDI notes must be 0-127.")
        self.out.send(mido.Message('note_off', note=note, velocity=velocity, channel=0))
    
    def play_voice_command(self, command: VoiceCommand):
        """
        Play a voice command by sending it as a MIDI note.
        
        Voice commands use MIDI note values 0-59 (values 60+ are reserved
        for regular piano notes). The command is sent as a note_on followed
        immediately by note_off to trigger the voice playback on the Teensy.
        """
        note_value = command.value
        
        # MIDI note values are strictly 0-127
        if note_value > 127:
            raise ValueError(
                f"VoiceCommand {command.name} has value {note_value} which exceeds "
                f"MIDI note range (0-127). Cannot send via standard MIDI protocol."
            )
        
        # Voice commands should be in range 0-59 (values 60+ are for regular notes)
        if note_value >= 60:
            import logging
            logging.warning(
                f"VoiceCommand {command.name} has value {note_value} which is in "
                f"the regular MIDI note range (60+). Voice commands should be 0-59."
            )
        
        self.out.send(mido.Message('note_on', note=note_value, velocity=127, channel=0))
        self.out.send(mido.Message('note_off', note=note_value, velocity=0, channel=0))




def main():
    # example usage
    left_glove = GloveSerialManager(source_hand='L', port=LEFT_PORT, play_mode=PlayMode.FREEPLAY_MODE)
    right_glove = GloveSerialManager(source_hand='R', port=RIGHT_PORT, play_mode=PlayMode.FREEPLAY_MODE)
    audio_board = AudioSerialManager(port=AUDIO_PORT)

    result = left_glove.connect()
    while not result:
        time.sleep(1)
        result = left_glove.connect()

    result = right_glove.connect()
    while not result:
        time.sleep(1)
        result = right_glove.connect()

    result = audio_board.connect(exclude_ports=[LEFT_PORT, RIGHT_PORT])
    while not result:
        time.sleep(1)
        result = audio_board.connect(exclude_ports=[LEFT_PORT, RIGHT_PORT])

    if result:
        left_glove.start()
        right_glove.start()
        audio_board.start()


    while True:
        time.sleep(1)
        left_responses = left_glove.get_responses()
        if left_responses:
            for response in left_responses:
                instruction = left_glove.handle_line_rx(response)
                if instruction:
                    # check finger position on cv
                    pass

        right_responses = right_glove.get_responses()
        if right_responses:
            for response in right_responses:
                right_glove.handle_line_rx(response)
        
# both serial listeners
    # left_thread = threading.Thread(target=read_from_teensy, args=(LEFT_PORT, 'L'))
    # right_thread = threading.Thread(target=read_from_teensy, args=(RIGHT_PORT, 'R'))

    # left_thread.start()
    # right_thread.start()
