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
from typing import Optional, List, Dict, Callable
from pathlib import Path

from .protocols import GloveProtocol, AudioProtocol


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerialManager:
    """
    Manages serial communication with multiple devices using threads.
    
    Supports:
    - Glove controller: haptic feedback
    - Audio board: MIDI playback
    """
    
    def __init__(self, glove_port: Optional[str] = None, 
                 audio_port: Optional[str] = None,
                 baud_rate: int = 115200,
                 auto_connect: bool = True):
        """
        Initialize serial manager.
        
        Args:
            glove_port: Serial port for glove controller (None for auto-detect)
            audio_port: Serial port for audio board (None for auto-detect)
            baud_rate: Baud rate for serial communication
            auto_connect: If True, automatically connect on init
        """
        self.baud_rate = baud_rate
        self.glove_port = glove_port
        self.audio_port = audio_port
        
        # Serial connections
        self.glove_conn: Optional[serial.Serial] = None
        self.audio_conn: Optional[serial.Serial] = None
        
        # Queues for non-blocking I/O
        self.glove_send_queue = queue.Queue(maxsize=100)
        self.audio_send_queue = queue.Queue(maxsize=100)
        
        self.glove_recv_queue = queue.Queue(maxsize=100)
        self.audio_recv_queue = queue.Queue(maxsize=100)
        
        # Threads
        self._glove_send_thread: Optional[threading.Thread] = None
        self._glove_recv_thread: Optional[threading.Thread] = None
        self._audio_send_thread: Optional[threading.Thread] = None
        self._audio_recv_thread: Optional[threading.Thread] = None
        
        self._running = False
        
        # Callbacks for incoming messages
        self.glove_callback: Optional[Callable] = None
        self.audio_callback: Optional[Callable] = None
        
        if auto_connect:
            self.connect_all()
    
    def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all devices.
        
        Returns:
            Dictionary mapping device names to connection status
        """
        results = {}
        
        # Connect to glove controller
        if self.glove_port:
            results['glove'] = self._connect_glove(self.glove_port)
        else:
            results['glove'] = self._auto_connect_glove()
        
        # Connect to audio board
        if self.audio_port:
            results['audio'] = self._connect_audio(self.audio_port)
        else:
            results['audio'] = self._auto_connect_audio()
        
        if any(results.values()):
            self.start()
        
        return results
    
    def _connect_glove(self, port: str) -> bool:
        """Connect to glove controller on specified port."""
        try:
            self.glove_conn = serial.Serial(port, self.baud_rate, timeout=0.1)
            logger.info(f"Connected to glove controller on {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to glove on {port}: {e}")
            return False
    
    def _connect_audio(self, port: str) -> bool:
        """Connect to audio board on specified port."""
        try:
            self.audio_conn = serial.Serial(port, self.baud_rate, timeout=0.1)
            logger.info(f"Connected to audio board on {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to audio on {port}: {e}")
            return False
    
    def _auto_connect_glove(self) -> bool:
        """Auto-detect and connect to glove controller."""
        available_ports = self._list_serial_ports()
        logger.info(f"Available serial ports: {available_ports}")
        
        for port in available_ports:
            if self._test_port_for_glove(port):
                self.glove_port = port
                return self._connect_glove(port)
        
        logger.warning("Could not auto-detect glove controller")
        return False
    
    def _auto_connect_audio(self) -> bool:
        """Auto-detect and connect to audio board."""
        available_ports = self._list_serial_ports()
        
        for port in available_ports:
            if port == self.glove_port:
                continue  # Don't use same port twice
            if self._test_port_for_audio(port):
                self.audio_port = port
                return self._connect_audio(port)
        
        logger.warning("Could not auto-detect audio board")
        return False
    
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
    
    def _test_port_for_glove(self, port: str) -> bool:
        """Test if port is glove controller."""
        # TODO: Implement device identification logic
        # For now, return True if port is open
        try:
            s = serial.Serial(port, self.baud_rate, timeout=0.1)
            s.close()
            return True
        except:
            return False
    
    def _test_port_for_audio(self, port: str) -> bool:
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
        if self._running:
            return
        
        self._running = True
        
        # Glove controller threads
        if self.glove_conn:
            self._glove_send_thread = threading.Thread(
                target=self._glove_send_worker, daemon=True)
            self._glove_recv_thread = threading.Thread(
                target=self._glove_recv_worker, daemon=True)
            self._glove_send_thread.start()
            self._glove_recv_thread.start()
        
        # Audio board threads
        if self.audio_conn:
            self._audio_send_thread = threading.Thread(
                target=self._audio_send_worker, daemon=True)
            self._audio_recv_thread = threading.Thread(
                target=self._audio_recv_worker, daemon=True)
            self._audio_send_thread.start()
            self._audio_recv_thread.start()
        
        logger.info("Serial manager started")
    
    def stop(self):
        """Stop all communication threads."""
        self._running = False
        
        if self._glove_send_thread:
            self._glove_send_thread.join(timeout=1)
        if self._glove_recv_thread:
            self._glove_recv_thread.join(timeout=1)
        if self._audio_send_thread:
            self._audio_send_thread.join(timeout=1)
        if self._audio_recv_thread:
            self._audio_recv_thread.join(timeout=1)
        
        if self.glove_conn:
            self.glove_conn.close()
        if self.audio_conn:
            self.audio_conn.close()
        
        logger.info("Serial manager stopped")
    
    def send_glove_command(self, motor_id: int, midi_note: int, action: int):
        """
        Send command to glove controller (non-blocking).
        
        Args:
            motor_id: Motor ID (0-4)
            midi_note: MIDI note number
            action: Action code
        """
        message = GloveProtocol.pack(motor_id, midi_note, action)
        try:
            self.glove_send_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Glove send queue full, dropping message")
    
    def send_audio_note_on(self, note: int, velocity: int = 100):
        """
        Send note-on to audio board (non-blocking).
        
        Args:
            note: MIDI note number
            velocity: Velocity/volume (0-127)
        """
        message = AudioProtocol.pack_note_on(note, velocity)
        try:
            self.audio_send_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Audio send queue full, dropping message")
    
    def send_audio_note_off(self, note: int):
        """Send note-off to audio board."""
        message = AudioProtocol.pack_note_off(note)
        try:
            self.audio_send_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Audio send queue full, dropping message")
    
    def get_glove_responses(self) -> List[bytes]:
        """Get all pending glove responses."""
        responses = []
        while not self.glove_recv_queue.empty():
            try:
                responses.append(self.glove_recv_queue.get_nowait())
            except queue.Empty:
                break
        return responses
    
    def get_audio_responses(self) -> List[bytes]:
        """Get all pending audio responses."""
        responses = []
        while not self.audio_recv_queue.empty():
            try:
                responses.append(self.audio_recv_queue.get_nowait())
            except queue.Empty:
                break
        return responses
    
    def set_glove_callback(self, callback: Callable):
        """Set callback for incoming glove messages."""
        self.glove_callback = callback
    
    def set_audio_callback(self, callback: Callable):
        """Set callback for incoming audio messages."""
        self.audio_callback = callback
    
    # Worker threads
    
    def _glove_send_worker(self):
        """Worker thread for sending to glove controller."""
        while self._running:
            try:
                message = self.glove_send_queue.get(timeout=0.1)
                if self.glove_conn:
                    self.glove_conn.write(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending glove message: {e}")
    
    def _glove_recv_worker(self):
        """Worker thread for receiving from glove controller."""
        while self._running:
            try:
                if self.glove_conn and self.glove_conn.in_waiting > 0:
                    data = self.glove_conn.read(self.glove_conn.in_waiting)
                    self.glove_recv_queue.put_nowait(data)
                    if self.glove_callback:
                        self.glove_callback(data)
            except queue.Full:
                logger.warning("Glove recv queue full, dropping data")
            except Exception as e:
                logger.error(f"Error receiving glove message: {e}")
                time.sleep(0.1)
    
    def _audio_send_worker(self):
        """Worker thread for sending to audio board."""
        while self._running:
            try:
                message = self.audio_send_queue.get(timeout=0.1)
                if self.audio_conn:
                    self.audio_conn.write(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending audio message: {e}")
    
    def _audio_recv_worker(self):
        """Worker thread for receiving from audio board."""
        while self._running:
            try:
                if self.audio_conn and self.audio_conn.in_waiting > 0:
                    data = self.audio_conn.read(self.audio_conn.in_waiting)
                    self.audio_recv_queue.put_nowait(data)
                    if self.audio_callback:
                        self.audio_callback(data)
            except queue.Full:
                logger.warning("Audio recv queue full, dropping data")
            except Exception as e:
                logger.error(f"Error receiving audio message: {e}")
                time.sleep(0.1)
    
    def is_connected(self) -> Dict[str, bool]:
        """Check connection status of all devices."""
        return {
            'glove': self.glove_conn is not None and self.glove_conn.is_open,
            'audio': self.audio_conn is not None and self.audio_conn.is_open
        }

