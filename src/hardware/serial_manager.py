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
import mido
from mido.backends import rtmidi
mido.set_backend('mido.backends.rtmidi')

from typing import Optional, List, Dict, Callable, Literal, Union
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


from .protocols import GloveProtocolFreeplayMode, GloveProtocolLearningMode, PlayingMode, Hand, SensorValue, VoiceCommand

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
            ports = [p for p in ports if "usbmodem" in p or "usbserial" in p] #disincludes bluetooth ports on mac
        else:
            ports = []
        
        available = []
        for port in ports:
            try:
                s = serial.Serial(port, timeout=0.1)
                s.close()
                available.append(port)
            except:
                pass
        
        return available
    
    def _connect(self, port: str, max_retries: int = 5) -> tuple[bool, Optional[Hand]]:
        """
        Connect to device on specified port with robust handshake.
        
        Args:
            port: Serial port to connect to
            max_retries: Maximum number of connection attempts
            
        Returns:
            Tuple of (success: bool, detected_hand: Optional[Hand])
        """
        HANDSHAKE_REQUEST = 0xAA
        HANDSHAKE_ACK = 0x55
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{max_retries} on {port}")
                
                # Step 1: Open serial port with longer timeout
                self.conn = serial.Serial(
                    port, 
                    baudrate=self.baud_rate, 
                    timeout=1.0,
                    write_timeout=1.0,
                    parity=serial.PARITY_NONE
                )
                
                logger.info(f"Clearing input and output buffers for {port}")

                # Step 2: Clear any stale data
                time.sleep(0.1)  # Let any boot messages arrive
                self.conn.reset_input_buffer()
                self.conn.reset_output_buffer()
                time.sleep(0.1)
                
                logger.info(f"Sending handshake request to {port}")

                # Step 3: Send handshake request
                self.conn.write(bytes([HANDSHAKE_REQUEST]))
                self.conn.flush()  # Ensure data is sent
                
                # Step 4: Wait for ACK
                start_time = time.time()
                ack_received = False
                
                while time.time() - start_time < 2.0:  # 2 second timeout for ACK
                    if self.conn.in_waiting > 0:
                        ack_byte = self.conn.read(1)
                        if ack_byte and ack_byte[0] == HANDSHAKE_ACK:
                            ack_received = True
                            logger.info(f"Handshake ACK received on {port}")
                            break
                        elif ack_byte and ack_byte[0] != HANDSHAKE_ACK:
                            logger.warning(f"Invalid ACK byte: {ack_byte.decode()} on {port}")
                            print(ack_byte)

                            
                    time.sleep(0.01)
                
                if not ack_received:
                    logger.warning(f"No handshake ACK on {port}, retrying...")
                    self.conn.close()
                    time.sleep(0.5)
                    continue
                
                # Step 5: Read hand identifier
                hand_bytes = self.conn.read(1)
                if not hand_bytes:
                    logger.warning(f"No hand byte received on {port}, retrying...")
                    self.conn.close()
                    time.sleep(0.5)
                    continue
                
                hand_value = hand_bytes[0]
                logger.info(f"Received hand byte: {hand_value} on {port}")
                
                # Step 6: Validate hand byte
                detected_hand = None
                if hand_value == Hand.LEFT.value:
                    logger.info(f"Detected left hand on {port}")
                    detected_hand = Hand.LEFT
                elif hand_value == Hand.RIGHT.value:
                    logger.info(f"Detected right hand on {port}")
                    detected_hand = Hand.RIGHT
                else:
                    logger.warning(f"Invalid hand value: {hand_value} on {port}, retrying...")
                    self.conn.close()
                    time.sleep(0.5)
                    continue
                
                # Step 7: Send confirmation
                self.conn.write(bytes([HANDSHAKE_ACK]))
                self.conn.flush()
                
                # Step 8: Clear buffers one final time
                time.sleep(0.1)
                self.conn.reset_input_buffer()
                
                self.port = port
                logger.info(f"Successfully connected to {detected_hand.name} hand on {port}")
                return True, detected_hand
                
            except serial.SerialException as e:
                logger.error(f"Serial exception on {port} (attempt {attempt + 1}): {e}")
                self.disconnect()

                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Unexpected error on {port} (attempt {attempt + 1}): {e}")
                self.disconnect()
                time.sleep(0.5)
        
        logger.error(f"Failed to connect on {port} after {max_retries} attempts")
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
    
    def _auto_connect(self, num_retries: int = 5) -> tuple[bool, Optional[Hand]]:
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
                success, detected_hand = self._connect(port, num_retries)
                if success:
                    return success, detected_hand
            else:
                logger.warning(f"Could not test port {port}")
                continue
        
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

        self.audio_board = None


    
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
        if hand == Hand.LEFT.value:
            return LeftGloveSerialManager(port=port, baud_rate=baud_rate)
        elif hand == Hand.RIGHT.value:
            return RightGloveSerialManager(port=port, baud_rate=baud_rate)
        else:
            raise ValueError(f"Invalid hand: {hand.name}")
    
    def connect(self, num_retries: int = 5) -> tuple[bool, Optional[Hand], Optional[Union['LeftGloveSerialManager', 'RightGloveSerialManager']]]:
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
            success, detected_hand = self._connect(self.port, num_retries)
        else:
            success, detected_hand = self._auto_connect(num_retries)
        
        logger.info(f"Connection success: {success}, Detected hand: {detected_hand}")
        
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
            # Transfer the connection
            correct_manager.conn = self.conn
            correct_manager.port = self.port
            correct_manager.hand = detected_hand
            # Clear our connection so we don't close it
            self.conn = None
            # Start the correct manager
            correct_manager._start()
            return True, detected_hand, correct_manager
        
        # Correct hand detected, update self.hand and start
        self.hand = detected_hand
        self._start()
        return True, detected_hand, self
    
    def recieve_voice_command(self) -> None:
        """
        Recieve a voice command byte from the glove controller.
        """
        logger.info(f"Recieving voice command for {self.hand}-hand glove controller")

        # Recieve voice command byte

        while self.conn.in_waiting == 0:
            time.sleep(0.1)
        voice_command = self.conn.read(1)
        self.conn.flush()
        print(f"Voice command byte received: {voice_command}")
        return voice_command[0]
    
    def receive_byte(self) -> Optional[int]:
        """
        Recieve a single byte from the glove controller.
        Blocking until a byte is received.
        """

        # Recieve byte

        while self.conn.in_waiting == 0:
            time.sleep(0.1)

        byte = self.conn.read(1)
        if byte != b'':
            self.conn.reset_input_buffer()
            number = byte[0]
            print(f"byte received: {number}")

        else: 
            print("No byte received")
            number = None


        return number


    
  

    def _start(self):
        """
        Start all communication threads.

        This will correspond to setup on the left glove controller.
        
        Note: This consumes the mode byte from the serial stream.
        If the glove only sends this byte once on startup, ensure it's
        available when start() is called.
        """
        if self._running or not self.conn:
            return
        
        

        return True

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

            if instruction.sensorValue == SensorValue.Pressed.value: # we can use this to then map to cv calls to assign a note and play it
                print(f"[{instruction.hand}] Sensor {instruction.sensorNumber} PRESSED") #instead of just printing out
                

            elif instruction.sensorValue == SensorValue.Released.value: # this we can use to map to a note and turn it off
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
            action: Action code -MAY REFER TO OLD ACTION CODES TALK TO NIKK - RAJUL AND SKY 
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
    
    def _start(self):
        """
        Start all communication threads.

        This will correspond to setup on the right glove controller.
        
        Note: This consumes the mode byte from the serial stream.
        If the glove only sends this byte once on startup, ensure it's
        available when start() is called.
        """
        super()._start()  # Call parent implementation






class AudioBoardManager:
    
    def __init__(self, port: str = None):
        """
        Initialize audio board manager.
        
        Args:
            port: Serial port for audio board (None for auto-detect)
        """
        # TODO: handshake with audio board?
        logger.info(f"Initializing audio board manager on port: {port}")
        self.port = None  # Initialize to avoid AttributeError
        self._is_connected = False


        if port:
            self.port = port
        else:

            print(mido.get_output_names())
            available_ports = mido.get_output_names()
            logger.info(f"Available audio ports: {available_ports}")
            if len(available_ports) == 0:
                raise ValueError("No audio ports found")
            
            for port in available_ports:
                if "Teensy MIDI" in port:
                    self.port = port
                    break
            
            if not self.port:
                raise ValueError("No Teensy MIDI port found")

        # open mido port
        self.out = mido.open_output(self.port)
        if not self.out:
            raise ValueError(f"Failed to open MIDI output port: {self.port}")

        self._is_connected = True
        logger.info(f"Audio board manager initialized on port: {self.port}")

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
        Voice commands will assume delays on left glove
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
            logger.warning(
                f"VoiceCommand {command.name} has value {note_value} which is in "
                f"the regular MIDI note range (60+). Voice commands should be 0-59."
            )
        
        self.out.send(mido.Message('note_on', note=note_value, velocity=127, channel=0))
        
        # TODO: figure out how to not wait for 10 fucking seconds

    def is_connected(self) -> bool:
        return self._is_connected





