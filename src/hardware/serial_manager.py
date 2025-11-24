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
from typing import Optional, List, Dict, Callable, Literal
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


from .protocols import GloveProtocolFreeplayMode, GloveProtocolLearningMode, AudioProtocol, PlayMode, Hand, SensorValue, SensorNumberLeft

from src.core.constants import SERIAL_BAUD_RATE, LEFT_PORT, RIGHT_PORT


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    def _connect(self, port: str) -> bool:
        """Connect to device on specified port."""
        try:
            self.conn = serial.Serial(port, self.baud_rate, timeout=0.1)
            self.port = port
            logger.info(f"Connected to device on {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect on {port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from device."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def is_connected(self) -> bool:
        """Check if connected to device."""
        return self.conn is not None and self.conn.is_open
    
    def stop(self):
        """Stop all communication threads."""
        self._running = False
        self.disconnect()


class GloveSerialManager(BaseSerialManager):
    """
    Manages serial communication with glove controller for haptic feedback.
    """
    
    def __init__(self, 
                 source_hand: Literal['L', 'R'],
                 port: Optional[str] = None, 

                 play_mode: PlayMode = PlayMode.FREEPLAY_MODE,
                 baud_rate: int = 115200, 
                ):
        """
        Initialize glove serial manager.
        
        Args:
            port: Serial port for glove controller (None for auto-detect)
            source_hand: 'L' or 'R' from which hand the glove controller is connected to
            baud_rate: Baud rate for serial communication
            auto_connect: If True, automatically connect on init
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
        
        self.hand = source_hand
        self._play_mode = play_mode

        
    
    def connect(self) -> bool:
        """
        Connect to glove controller.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.port:
            result = self._connect(self.port)
        else:
            result = self._auto_connect()
        
        if result:
            self.start()
        
        return result
    
    def _auto_connect(self) -> bool:
        """Auto-detect and connect to glove controller."""
        available_ports = self._list_serial_ports()
        logger.info(f"Available serial ports: {available_ports}")
        
        for port in available_ports:
            if self._test_port(port):
                return self._connect(port)
        
        logger.warning("Could not auto-detect glove controller")
        return False
    
    def _test_port(self, port: str) -> bool:
        """Test if port is glove controller."""
        # TODO: Test device identification logic
        # For now, return True if port is open
        try:
            s = serial.Serial(port, self.baud_rate, timeout=0.1)
            glove_hand = s.read(1)
            if glove_hand == Hand.LEFT & self.hand == 'L':
                logger.info(f"Found {self.hand}-hand glove controller on {port}")
                s.close()
                return True
            elif glove_hand == Hand.RIGHT & self.hand == 'R':
                logger.info(f"Found correct {self.hand}-hand glove controller on {port}")
                s.close()
                return True
            else:
                logger.warning(f"Found incorrect {glove_hand}-hand glove controller on {port}, switching hand info...")
                s.close()
                self.hand = glove_hand
                return False
        except:
            return False
    
    def start(self):
        """Start all communication threads."""
        if self._running or not self.conn:
            return
        
        if self.conn.in_waiting > 0:
            # read the first byte in stream to get the mode info
            mode = self.conn.read(1)
            if mode == PlayMode.LEARNING_MODE:
                self._play_mode = PlayMode.LEARNING_MODE
            elif mode == PlayMode.FREEPLAY_MODE:
                self._play_mode = PlayMode.FREEPLAY_MODE
            else:
                raise ValueError(f"Invalid play mode: {mode}")

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

        if self._play_mode == PlayMode.FREEPLAY_MODE:
            instruction = GloveProtocolFreeplayMode.unpack(line)
            if instruction.hand != self.hand:
                logger.warning(f"Found incorrect {instruction.hand}-hand glove controller on {self.port}, switching hand info...")
                self.hand = instruction.hand
                return None        

            if instruction.sensorValue == SensorValue.Pressed: # we can use this to then map to cv calls to assign a note and play it
                print(f"[{instruction.hand}] Sensor {instruction.sensorNumber} PRESSED") #instead of just printing out
                

            elif instruction.sensorValue == SensorValue.Released: # this we can use to map to a note and turn it off
                print(f"[{instruction.hand}] Sensor {instruction.sensorNumber} RELEASED") #instead of just printing it otu

        elif self._play_mode == PlayMode.LEARNING_MODE:
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
        if self._play_mode == PlayMode.FREEPLAY_MODE:
            # TODO: this is wrong for freeplay mode
            message = GloveProtocolFreeplayMode.pack(motor_id, midi_note, action)
        elif self._play_mode == PlayMode.LEARNING_MODE:
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


#TODO: audio serial manager should use MIDO
class AudioSerialManager(BaseSerialManager):
    """
    Manages serial communication with audio board for MIDI playback.
    """
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200,
                 auto_connect: bool = True):
        """
        Initialize audio serial manager.
        
        Args:
            port: Serial port for audio board (None for auto-detect)
            baud_rate: Baud rate for serial communication
            auto_connect: If True, automatically connect on init
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
        
        if port is None:
            self.connect()
    
    def connect(self, exclude_ports: Optional[List[str]] = None) -> bool:
        """
        Connect to audio board.
        
        Args:
            exclude_ports: List of ports to exclude from auto-detection
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.port:
            result = self._connect(self.port)
        else:
            result = self._auto_connect(exclude_ports=exclude_ports)
        
        if result:
            self.start()
        
        return result
    
    def _auto_connect(self, exclude_ports: Optional[List[str]] = None) -> bool:
        """Auto-detect and connect to audio board."""
        available_ports = self._list_serial_ports()
        
        if exclude_ports is None:
            exclude_ports = []
        
        for port in available_ports:
            if port in exclude_ports:
                continue
            if self._test_port(port):
                return self._connect(port)
        
        logger.warning("Could not auto-detect audio board")
        return False
    
    def _test_port(self, port: str) -> bool:
        """Test if port is audio board."""
        # TODO: Implement device identification logic
        try:
            s = serial.Serial(port, self.baud_rate, timeout=0.1)
            s.close()
            return True
        except:
            return False
    
    def start(self):
        """Start all communication threads."""
        if self._running or not self.conn:
            return
        
        self._running = True
        
        self._send_thread = threading.Thread(
            target=self._send_worker, daemon=True)
        self._recv_thread = threading.Thread(
            target=self._recv_worker, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()
        
        logger.info("Audio serial manager started")
    
    def stop(self):
        """Stop all communication threads."""
        super().stop()
        
        if self._send_thread:
            self._send_thread.join(timeout=1)
        if self._recv_thread:
            self._recv_thread.join(timeout=1)
        
        logger.info("Audio serial manager stopped")
    
    def send_note_on(self, note: int, velocity: int = 100):
        """
        Send note-on to audio board (non-blocking).
        
        Args:
            note: MIDI note number
            velocity: Velocity/volume (0-127)
        """
        message = AudioProtocol.pack_note_on(note, velocity)
        try:
            self.send_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Audio send queue full, dropping message")
    
    def send_note_off(self, note: int):
        """Send note-off to audio board."""
        message = AudioProtocol.pack_note_off(note)
        try:
            self.send_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Audio send queue full, dropping message")
    
    def get_responses(self) -> List[bytes]:
        """Get all pending audio responses."""
        responses = []
        while not self.recv_queue.empty():
            try:
                responses.append(self.recv_queue.get_nowait())
            except queue.Empty:
                break
        return responses
    
    def set_callback(self, callback: Callable):
        """Set callback for incoming audio messages."""
        self.callback = callback
    
    def _send_worker(self):
        """Worker thread for sending to audio board."""
        while self._running:
            try:
                message = self.send_queue.get(timeout=0.1)
                if self.conn:
                    self.conn.write(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending audio message: {e}")
    
    def _recv_worker(self):
        """Worker thread for receiving from audio board."""
        while self._running:
            try:
                if self.conn and self.conn.in_waiting > 0:
                    data = self.conn.read(self.conn.in_waiting)
                    self.recv_queue.put_nowait(data)
                    if self.callback:
                        self.callback(data)
            except queue.Full:
                logger.warning("Audio recv queue full, dropping data")
            except Exception as e:
                logger.error(f"Error receiving audio message: {e}")
                time.sleep(0.1)



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
