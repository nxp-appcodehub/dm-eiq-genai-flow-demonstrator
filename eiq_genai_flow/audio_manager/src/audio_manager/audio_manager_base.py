# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

"""
Audio Manager Base Module

Provides abstract base class and common functionality for audio capture/playback
with support for multiple readers, recording, and format conversion.

Architecture Overview:
---------------------
    ┌───────────────────────────────────────────────────────────┐
    │                    Audio Manager                          │
    │                                                           │
    │  ┌──────────────┐         ┌─────────────────────┐         │
    │  │   Capture    │────────>│  Master Buffer      │         │
    │  │   Device     │         │  (Circular Deque)   │         │
    │  └──────────────┘         └─────────────────────┘         │
    │                                    │                      │
    │                           ┌────────┴────────┐             │
    │                           │                 │             │
    │                    ┌──────▼──────┐   ┌──────▼──────┐      │
    │                    │  Reader 1   │   │  Reader 2   │      │
    │                    │  (VIT)      │   │  (STT)      │      │
    │                    └─────────────┘   └─────────────┘      │
    │                                                           │
    │  ┌──────────────┐         ┌─────────────────────┐         │
    │  │  Playback    │<────────│  Playback Queue     │         │
    │  │   Device     │         │  (Deque)            │         │
    │  └──────────────┘         └─────────────────────┘         │
    └───────────────────────────────────────────────────────────┘

Key Concepts:
------------
1. **Master Buffer**: Circular buffer storing captured audio samples
   - All readers read from this shared buffer
   - Each reader maintains independent read pointer
   - Automatically discards old data when full

2. **Multi-Reader Pattern**: Multiple consumers can read the same audio stream
   - Each reader has independent position tracking
   - Format/channel conversion per reader
   - Event-driven blocking reads available

3. **Thread Safety**: All operations are protected by locks
   - master_buffer_lock: Protects capture buffer and write_index
   - playback_queue_lock: Protects playback queue
   - readers_lock: Protects reader registry
   - Condition variables for efficient blocking

4. **Format Conversion**: Automatic audio format conversion
   - Master buffer stores in capture format
   - Readers get data in their requested format
   - Supports: S32LE, S16LE, F32LE

5. **Channel Extraction**: Flexible channel routing
   - Readers can extract specific channels from multi-channel stream
   - Mono/stereo conversion as needed
"""

import os
import wave
import logging
import threading
import numpy as np
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict
from audio_manager.set_audio_device_config import set_capture_device_config, set_playback_device_config

logger = logging.getLogger(__name__)

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning("soundfile not available - float WAV recording will fall back to int32")


@dataclass
class CaptureConfig:
    """
    Audio capture configuration.

    Attributes:
        capture_device: ALSA device name (e.g., "default", "plughw:CARD=wm8962audio")
        sample_rate: Sampling rate in Hz (typically 16000)
        channels: Number of capture channels (e.g., 2 for stereo)
        format: Audio format - S32LE (32-bit int), S16LE (16-bit int), F32LE (32-bit float)
        frame_duration_ms: Duration of each audio frame in milliseconds
        buffer_duration_sec: Total duration of master buffer in seconds
        save_audio: Enable recording to WAV file
        audio_save_path: Directory for saved audio files
        keep_device_open: Keep capture device open continuously (vs on-demand)
    """

    capture_device: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 2
    format: str = "S32LE"  # S32LE, S16LE, F32LE
    frame_duration_ms: int = 30
    buffer_duration_sec: int = 30
    save_audio: bool = False
    audio_save_path: str = "."
    keep_device_open: bool = True

    @property
    def frame_duration_sec(self) -> float:
        """Frame duration in seconds (derived from ms)."""
        return self.frame_duration_ms / 1000.0

    @property
    def samples_per_frame(self) -> int:
        """Number of samples per audio frame."""
        return int(self.sample_rate * self.frame_duration_sec)

    @property
    def bytes_per_sample(self) -> int:
        """Bytes per sample based on format."""
        format_bytes = {"S32LE": 4, "S32BE": 4, "F32LE": 4, "F32BE": 4, "S16LE": 2, "S16BE": 2, "S8": 1, "U8": 1}
        return format_bytes.get(self.format, 4)

    @property
    def blocksize_bytes(self) -> int:
        """Total blocksize in bytes for hardware reads."""
        return int(self.sample_rate * self.channels * self.bytes_per_sample * self.frame_duration_sec)


@dataclass
class PlaybackConfig:
    """
    Audio playback configuration.

    Attributes:
        playback_device: ALSA device name
        sample_rate: Default playback sample rate (can be changed dynamically)
        channels: Output channels (1=mono, 2=stereo)
        format: Audio format
        frame_duration_ms: Frame duration for timing calculations
        save_audio: Enable recording of played audio to WAV file
        audio_save_path: Directory for saved audio files
        keep_device_open: Keep device open for low latency (vs re-open per chunk)
    """

    playback_device: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1
    format: str = "S32LE"
    frame_duration_ms: int = 30
    save_audio: bool = False
    audio_save_path: str = "."
    keep_device_open: bool = False

    @property
    def frame_duration_sec(self) -> float:
        """Frame duration in seconds."""
        return self.frame_duration_ms / 1000.0

    @property
    def samples_per_frame(self) -> int:
        """Number of samples per audio frame."""
        return int(self.sample_rate * self.frame_duration_sec)


@dataclass
class ReaderConfig:
    """
    Configuration for an audio reader.

    Each reader can request a specific format and channel configuration,
    independent of the master buffer's format.

    Attributes:
        channels: Number of output channels for this reader
        format: Desired audio format (will be converted from master format)
        channel_indices: Which master buffer channels to extract
                        e.g., [0] = first channel only
                             [0, 1] = first two channels

    Example:
        # Extract mono from first channel, convert to float32
        ReaderConfig(channels=1, format="F32LE", channel_indices=[0])

        # Extract stereo from channels 0 and 1, keep as int16
        ReaderConfig(channels=2, format="S16LE", channel_indices=[0, 1])
    """

    channels: int = 1
    format: str = "S32LE"
    channel_indices: Optional[list] = None

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype for this format."""
        dtype_map = {"F32LE": np.float32, "S32LE": np.int32, "S16LE": np.int16}
        return dtype_map.get(self.format, np.float32)

    def __post_init__(self):
        """Set default channel indices if not provided."""
        if self.channel_indices is None:
            self.channel_indices = list(range(self.channels))
        elif len(self.channel_indices) != self.channels:
            raise ValueError("channel_indices length must match channels count")


class AudioReader:
    """
    Independent audio stream reader with callback support.

    Each reader maintains its own position in the master buffer and can:
    - Read audio at its own pace
    - Get audio in its preferred format/channels
    - Block until data is available (event-driven)
    - Skip or re-read portions of the buffer

    Thread Safety:
        All operations are thread-safe through the manager's locks.

    Usage:
        reader = manager.register_reader("MyReader", ReaderConfig(channels=1, format="F32LE"))
        reader.enable(sync_to_current=True)

        while True:
            samples = reader.read(512, blocking=True)  # Event-driven blocking
            if samples is not None:
                process(samples)
    """

    def __init__(self, manager: "AudioManager", name: str, config: ReaderConfig, callback: Optional[Callable] = None):
        """
        Initialize audio reader.

        Args:
            manager: Parent AudioManager instance
            name: Unique identifier for this reader
            config: Reader configuration (format, channels, etc.)
            callback: Optional callback function called on new audio (not yet implemented)
        """
        self.manager = manager
        self.name = name
        self.config = config
        self.callback = callback
        self.enabled = False
        self.read_index = 0  # Absolute position in master buffer

    def enable(self, sync_to_current: bool = False):
        """
        Enable this reader and set initial read position.

        Args:
            sync_to_current: If True, start reading from the current write position
                           (real-time mode - skip buffered past data)
                           If False, start from the oldest available sample
                           (catch-up mode - read all buffered data)

        Thread Safety:
            Uses manager's master_buffer_lock for atomic position setting.
        """
        self.enabled = True
        with self.manager.master_buffer_lock:
            if sync_to_current:
                # Real-time: Start from current position (skip past data)
                self.sync_to_current()
            else:
                # Catch-up: Start from oldest available sample
                self.read_index = self.manager.write_index - len(self.manager.master_buffer)
        logger.debug(f"Reader '{self.name}' enabled ({self.config.channels}ch {self.config.format})")

    def sync_to_current(self):
        self.read_index = self.manager.write_index

    def disable(self):
        """Disable this reader (stops reading but doesn't unregister)."""
        self.enabled = False
        logger.debug(f"Reader '{self.name}' disabled")

    def unregister(self):
        """Unregister this reader from the audio manager completely."""
        self.manager.unregister_reader(self.name)
        logger.debug(f"Reader '{self.name}' unregistered")

    def read(self, num_samples: int, blocking: bool = False, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read samples from master buffer with format/channel conversion.

        Args:
            num_samples: Number of samples to read
            blocking: If True, wait for data using condition variable (event-driven)
                     If False, return None immediately if insufficient data
            timeout: Maximum time to wait if blocking (None = infinite)
                    Recommended: Set to slightly longer than frame duration

        Returns:
            NumPy array of audio data in reader's format, or None if:
            - Reader is disabled
            - Insufficient data available (non-blocking mode)
            - Timeout occurred (blocking mode)
            - Read pointer is invalid (fell behind buffer)

        Thread Safety:
            Uses manager's master_buffer_lock and new_data_cv for thread-safe access.

        Example:
            # Non-blocking (polling)
            samples = reader.read(512)
            if samples is None:
                # No data yet
                time.sleep(0.01)

            # Blocking (event-driven - recommended)
            samples = reader.read(512, blocking=True, timeout=0.1)
            # Wakes immediately when data arrives, or after 100ms timeout
        """
        if not self.enabled:
            return None

        if blocking:
            # Event-driven blocking: Wait for data using condition variable
            with self.manager.new_data_cv:
                while True:
                    # Try to read
                    result = self._try_read(num_samples)
                    if result is not None:
                        return result

                    # Wait for new data notification or timeout
                    if not self.manager.new_data_cv.wait(timeout=timeout):
                        # Timeout occurred
                        return None
        else:
            # Non-blocking mode
            return self._try_read(num_samples)

    def _try_read(self, num_samples: int) -> Optional[np.ndarray]:
        """
        Internal helper: Try to read samples without blocking.

        This method handles:
        1. Validating read pointer is within buffer bounds
        2. Checking if enough samples are available
        3. Extracting samples from circular buffer
        4. Advancing read pointer
        5. Converting format/channels for reader

        Returns:
            Audio data array or None if insufficient data
        """
        # Calculate valid buffer range
        buffer_start = self.manager.write_index - len(self.manager.master_buffer)

        # Check if read pointer is valid
        if self.read_index < buffer_start or self.read_index > self.manager.write_index:
            # Read pointer fell behind or went ahead - invalid position
            return None

        # Calculate offset and available samples
        offset = self.read_index - buffer_start
        available = len(self.manager.master_buffer) - offset

        if available < num_samples:
            # Not enough data available yet
            return None

        # Extract samples from circular buffer
        samples = [self.manager.master_buffer[i] for i in range(offset, offset + num_samples)]
        self.read_index += num_samples

        # Convert to array with master format's dtype
        master_format = self.manager._normalize_format(self.manager.capture_config.format)
        dtype = self.manager._get_dtype_for_format(master_format)
        audio_data = np.array(samples, dtype=dtype)

        # Convert to reader's requested format and channels
        return self.manager._convert_audio_for_reader(audio_data, self.config)


class AudioRecorder:
    """
    WAV file recorder supporting integer and float formats.

    Automatically handles:
    - Format conversion (int16, int32, float32)
    - Channel conversion (mono/stereo)
    - Timestamped filenames
    - Proper WAV file cleanup

    Uses soundfile library for float32 if available, falls back to wave module.

    Example:
        recorder = AudioRecorder("./recordings", 16000, channels=1, format="F32LE")
        recorder.start()
        recorder.write(audio_data)
        recorder.stop()  # File automatically saved
    """

    def __init__(
        self, save_directory: str, sample_rate: int, channels: int = 1, filename_prefix: str = "", format: str = "S16LE"
    ):
        """
        Initialize audio recorder.

        Args:
            save_directory: Directory to save WAV files
            sample_rate: Audio sample rate
            channels: Number of channels (1=mono, 2=stereo)
            filename_prefix: Prefix for generated filenames
            format: Audio format (S16LE, S32LE, F32LE)
        """
        self.save_directory = save_directory
        self.sample_rate = sample_rate
        self.channels = channels
        self.filename_prefix = filename_prefix
        self.format = format
        self.wav_file = None
        self.sf_file = None
        self.filepath = None
        self.is_recording = False
        self.use_soundfile = format in ["F32LE", "F32BE"] and HAS_SOUNDFILE

    def start(self):
        """
        Start recording to a new WAV file.

        Creates a timestamped filename and opens the file for writing.
        Directory is created if it doesn't exist.
        """
        try:
            os.makedirs(self.save_directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{self.filename_prefix}{timestamp}.wav"
            self.filepath = os.path.join(self.save_directory, filename)

            if self.use_soundfile:
                # Use soundfile for float formats (better quality)
                self.sf_file = sf.SoundFile(
                    self.filepath,
                    mode="w",
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    subtype="FLOAT",
                    format="WAV",
                )
                logger.info(f"Recording: {self.filepath} ({self.format}, {self.channels}ch) [soundfile]")
            else:
                # Use wave module for integer formats
                if self.format in ["F32LE", "F32BE"]:
                    logger.warning(f"soundfile unavailable, converting {self.format} to S32LE")
                    self.format = "S32LE"

                self.wav_file = wave.open(self.filepath, "wb")
                self.wav_file.setnchannels(self.channels)
                sample_width = {"S32LE": 4, "S16LE": 2, "S8": 1}.get(self.format, 2)
                self.wav_file.setsampwidth(sample_width)
                self.wav_file.setframerate(self.sample_rate)
                logger.info(f"Recording: {self.filepath} ({self.format}, {self.channels}ch) [wave]")

            self.is_recording = True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False

    def write(self, audio_data: np.ndarray):
        """
        Write audio data to file with automatic format conversion.

        Automatically handles:
        - Channel conversion (mono<->stereo)
        - Format conversion (int<->float)
        - Clipping to valid ranges

        Args:
            audio_data: Audio samples as NumPy array
                       Shape: (samples,) for mono or (samples, channels) for multi-channel
        """
        if not self.is_recording:
            return

        try:
            # Handle channel conversion
            if audio_data.ndim == 1 and self.channels > 1:
                # Mono to multi-channel: duplicate
                audio_data = np.stack([audio_data] * self.channels, axis=-1)
            elif audio_data.ndim == 2 and audio_data.shape[-1] != self.channels:
                if self.channels == 1:
                    # Multi-channel to mono: average
                    audio_data = np.mean(audio_data, axis=-1)
                else:
                    # Take first N channels
                    audio_data = audio_data[:, : self.channels]

            if self.use_soundfile and self.sf_file:
                # Convert to float32 for soundfile
                if audio_data.dtype != np.float32:
                    if audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    elif audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                audio_data = np.nan_to_num(np.clip(audio_data, -1.0, 1.0), nan=0.0)
                self.sf_file.write(audio_data)

            elif self.wav_file:
                # Convert to target integer format
                if self.format in ["S32LE", "S32BE"]:
                    if audio_data.dtype == np.float32:
                        audio_data = (np.clip(audio_data, -1.0, 1.0) * 2147483647).astype(np.int32)
                    elif audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.int32) << 16
                elif self.format in ["S16LE", "S16BE"]:
                    if audio_data.dtype == np.float32:
                        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                    elif audio_data.dtype == np.int32:
                        audio_data = (audio_data >> 16).astype(np.int16)

                self.wav_file.writeframes(audio_data.tobytes())

        except Exception as e:
            logger.error(f"Error writing audio: {e}")

    def stop(self):
        """Stop recording and close file properly."""
        try:
            if self.sf_file:
                self.sf_file.close()
                self.sf_file = None
            elif self.wav_file:
                self.wav_file.close()
                self.wav_file = None

            if self.is_recording:
                logger.info(f"Recording saved: {self.filepath}")
            self.is_recording = False
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")


class AudioManager(ABC):
    """
    Abstract base class for audio capture and playback.

    This class provides the common infrastructure for managing audio I/O with
    multiple simultaneous readers and playback. Backend implementations (ALSA,
    GStreamer) inherit from this and implement the abstract methods.

    Key Features:
    ------------
    1. **Multi-Reader Capture**
       - Multiple consumers can read the same audio stream independently
       - Each reader has its own position, format, and channel configuration
       - Thread-safe with event-driven blocking for efficiency

    2. **Circular Master Buffer**
       - Fixed-size buffer stores recent audio history
       - Automatically discards old samples when full
       - All readers read from this shared buffer

    3. **Format Conversion**
       - Master buffer stores audio in capture format
       - Readers automatically get data in their requested format
       - Supports S32LE ↔ S16LE ↔ F32LE conversions

    4. **Channel Routing**
       - Extract specific channels from multi-channel stream
       - Automatic mono/stereo conversion

    5. **Playback Queue**
       - Asynchronous playback with dynamic sample rate support
       - Queue-based for non-blocking operation
       - Supports multiple sample rates in same session

    6. **Recording**
       - Optional WAV file recording for both capture and playback
       - Automatic format conversion for recording

    Thread Safety:
    -------------
    All operations are protected by locks:
    - `master_buffer_lock`: Protects capture buffer and write_index
    - `new_data_cv`: Condition variable for event-driven reader blocking
    - `playback_queue_lock`: Protects playback queue
    - `playback_queue_cv`: Condition variable for playback thread synchronization
    - `readers_lock`: Protects reader registry

    Usage Pattern:
    -------------
        # Create manager
        manager = create_audio_manager(backend="alsa")

        # Register readers
        vit_reader = manager.register_reader("VIT", ReaderConfig(channels=1, format="S16LE"))
        stt_reader = manager.register_reader("STT", ReaderConfig(channels=1, format="F32LE"))

        # Start capture
        manager.start_capture()

        # Enable readers
        vit_reader.enable(sync_to_current=True)
        stt_reader.enable(sync_to_current=True)

        # Read audio (event-driven blocking)
        vit_samples = vit_reader.read(512, blocking=True, timeout=0.1)
        stt_samples = stt_reader.read(512, blocking=True, timeout=0.1)

        # Queue audio for playback
        manager.play_audio_async(audio_data, sample_rate=16000)

        # Wait for playback completion
        while not manager.is_playback_complete():
            time.sleep(0.1)

        # Cleanup
        manager.shutdown()
    """

    def __init__(
        self, capture_config: Optional[CaptureConfig] = None, playback_config: Optional[PlaybackConfig] = None
    ):
        """
        Initialize AudioManager base.

        Args:
            capture_config: Capture configuration (device, format, etc.)
            playback_config: Playback configuration (device, format, etc.)
        """
        self.capture_config = capture_config or CaptureConfig()
        self.playback_config = playback_config or PlaybackConfig()

        self.capture_device = self.capture_config.capture_device or "default"
        self.playback_device = self.playback_config.playback_device or "default"

        # ===== CAPTURE STATE =====
        self.capture_running = threading.Event()  # Flag: is capture active?
        self.capture_thread: Optional[threading.Thread] = None
        self.buffers_dropped = 0  # Drop first few buffers (transient noise)
        self.fade_in_done = False  # Apply fade-in to first valid buffer

        # ===== MASTER BUFFER (Circular Buffer) =====
        # This is the core shared buffer that all readers consume from
        max_buffer_len = self.capture_config.sample_rate * self.capture_config.buffer_duration_sec
        self.master_buffer = deque(maxlen=max_buffer_len)  # Auto-discards old samples
        self.write_index = 0  # Absolute position (monotonically increasing)
        self.master_buffer_lock = threading.Lock()  # Protects buffer and write_index
        self.new_data_cv = threading.Condition(self.master_buffer_lock)  # For event-driven reads

        # ===== READER MANAGEMENT =====
        self.readers: Dict[str, AudioReader] = {}  # name -> AudioReader
        self.readers_lock = threading.Lock()

        # ===== PLAYBACK STATE =====
        self.playback_queue = deque()  # Queue of (audio_data, sample_rate) tuples
        self.playback_queue_lock = threading.RLock()
        self.playback_queue_cv = threading.Condition(self.playback_queue_lock)
        self.playback_running = threading.Event()  # Flag: is playback thread running?
        self.playback_thread: Optional[threading.Thread] = None
        self.playback_active = threading.Event()  # Flag: is audio currently playing?
        self.stream_end_signaled = threading.Event()  # Flag: stream end has been signaled

        # ===== RECORDING SETUP =====
        self.audio_capture_recorder = None
        if self.capture_config.save_audio:
            self.audio_capture_recorder = AudioRecorder(
                self.capture_config.audio_save_path,
                self.capture_config.sample_rate,
                self.capture_config.channels,
                "capture_",
                format=self._normalize_format(self.capture_config.format),
            )

        self.audio_playback_recorder = None
        if self.playback_config.save_audio:
            self.audio_playback_recorder = AudioRecorder(
                self.playback_config.audio_save_path,
                self.playback_config.sample_rate,
                self.playback_config.channels,
                "playback_",
                format=self._normalize_format(self.playback_config.format),
            )
            self.audio_playback_recorder.start()

        # set capture device config
        set_capture_device_config(capture_config.capture_device)

        # set playback device config
        set_playback_device_config(playback_config.playback_device)

        logger.info(f"{self.__class__.__name__} initialized successfully")
        logger.debug(
            f"{self.__class__.__name__}: "
            f"Capture: {self.capture_config.sample_rate}Hz/{self.capture_config.channels}ch, "
            f"Playback: {self.playback_config.sample_rate}Hz/{self.playback_config.channels}ch"
        )

    # ===== ABSTRACT METHODS - Backend Implementation Required =====
    @abstractmethod
    def start_capture(self):
        """
        Start audio capture (backend-specific implementation).

        Backend should:
        1. Open audio capture device
        2. Start capture thread/callback
        3. Set capture_running flag
        4. Call _process_captured_audio() for each frame
        """
        pass

    @abstractmethod
    def stop_capture(self):
        """
        Stop audio capture (backend-specific implementation).

        Backend should:
        1. Clear capture_running flag
        2. Stop capture thread/callback
        3. Close audio device
        """
        pass

    @abstractmethod
    def start_playback(self):
        """
        Start playback thread (backend-specific implementation).

        Backend should:
        1. Set playback_running flag
        2. Start playback thread that processes playback_queue
        3. Optionally pre-open device if keep_device_open=True
        """
        pass

    @abstractmethod
    def stop_playback(self):
        """
        Stop playback thread (backend-specific implementation).

        Backend should:
        1. Clear playback_running flag
        2. Stop playback thread
        3. Close audio device
        """
        pass

    @abstractmethod
    def _playback_loop(self):
        """
        Playback thread loop (backend-specific implementation).

        Backend should:
        1. Loop while playback_running is set
        2. Pop audio from playback_queue (with locking)
        3. Call _push_audio_to_device()
        4. Set/clear playback_active flag appropriately
        5. Notify playback_queue_cv when done
        """
        pass

    @abstractmethod
    def _push_audio_to_device(self, audio_data: np.ndarray, sample_rate: int):
        """
        Push audio to hardware device (backend-specific implementation).

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate for this audio chunk

        Backend should:
        1. Convert audio_data to device format (if needed)
        2. Handle sample rate changes (reopen device if necessary)
        3. Write audio to device
        4. Block until audio is fully written (or use hardware buffering)

        Note:
            audio_data has already been prepared by _prepare_playback_audio()
            (normalized, clipped, channel-adjusted)
        """
        pass

    # ===== READER MANAGEMENT =====

    def register_reader(
        self, name: str, config: Optional[ReaderConfig] = None, callback: Optional[Callable] = None
    ) -> AudioReader:
        """
        Register a new audio reader with independent read pointer.

        This creates a new reader that can independently consume audio from the
        master buffer with its own format, channels, and position tracking.

        Args:
            name: Unique identifier for this reader (e.g., "VIT", "STT")
            config: Reader configuration (format, channels, channel indices)
            callback: Optional callback function (currently unused, reserved for future)

        Returns:
            AudioReader instance ready to be enabled

        Raises:
            ValueError: If reader requests more channels than available in master buffer

        Thread Safety:
            Protected by readers_lock

        Example:
            # Create reader for VIT (mono int16 from first channel)
            vit_reader = manager.register_reader(
                "VIT",
                ReaderConfig(channels=1, format="S16LE", channel_indices=[0])
            )

            # Create reader for STT (mono float32 from second channel)
            stt_reader = manager.register_reader(
                "STT",
                ReaderConfig(channels=1, format="F32LE", channel_indices=[1])
            )
        """
        with self.readers_lock:
            if name in self.readers:
                return self.readers[name]

            reader_config = config or ReaderConfig()

            # Validate reader channel configuration against master buffer
            if reader_config.channels > self.capture_config.channels:
                raise ValueError(
                    f"Reader '{name}' requests {reader_config.channels} channels, "
                    f"but master buffer only has {self.capture_config.channels} channels"
                )

            # Validate channel indices don't exceed available channels
            max_channel_index = max(reader_config.channel_indices)
            if max_channel_index >= self.capture_config.channels:
                raise ValueError(
                    f"Reader '{name}' requests channel index {max_channel_index}, "
                    f"but master buffer only has channels 0-{self.capture_config.channels - 1}"
                )

            reader = AudioReader(self, name, reader_config, callback)
            self.readers[name] = reader
            logger.debug(f"Registered reader '{name}' ({reader.config.channels}ch {reader.config.format})")
            return reader

    def unregister_reader(self, name: str):
        """
        Unregister and disable a reader.

        Args:
            name: Reader identifier

        Thread Safety:
            Protected by readers_lock
        """
        with self.readers_lock:
            if name in self.readers:
                self.readers[name].disable()
                del self.readers[name]

    def get_reader(self, name: str) -> Optional[AudioReader]:
        """
        Get reader by name.

        Args:
            name: Reader identifier

        Returns:
            AudioReader instance or None if not found

        Thread Safety:
            Protected by readers_lock
        """
        with self.readers_lock:
            return self.readers.get(name)

    # ===== FORMAT UTILITIES =====

    def _normalize_format(self, format_value) -> str:
        """
        Convert ALSA format constant to string if needed.

        Handles both string formats ("S32LE") and ALSA constants (aa.PCM_FORMAT_S32_LE)

        Args:
            format_value: String or ALSA format constant

        Returns:
            String format name (e.g., "S32LE", "S16LE", "F32LE")
        """
        if isinstance(format_value, str):
            return format_value

        # Handle ALSA format constants
        try:
            import alsaaudio as aa

            alsa_map = {
                aa.PCM_FORMAT_S32_LE: "S32LE",
                aa.PCM_FORMAT_S16_LE: "S16LE",
                aa.PCM_FORMAT_FLOAT_LE: "F32LE",
                aa.PCM_FORMAT_S8: "S8",
            }
            return alsa_map.get(format_value, "S32LE")
        except ImportError:
            return "S32LE"

    def _get_dtype_for_format(self, format_str: str) -> np.dtype:
        """
        Get NumPy dtype for format string.

        Args:
            format_str: Format string (e.g., "S32LE", "F32LE")

        Returns:
            NumPy dtype (np.int32, np.int16, or np.float32)
        """
        dtype_map = {"S32LE": np.int32, "S16LE": np.int16, "F32LE": np.float32}
        return dtype_map.get(format_str, np.float32)

    # ===== AUDIO CONVERSION =====
    def _convert_audio_for_reader(self, audio_data: np.ndarray, reader_config: ReaderConfig) -> np.ndarray:
        """
        Convert audio format and extract channels for reader.

        This performs two transformations:
        1. Channel extraction/conversion (based on channel_indices)
        2. Format conversion (e.g., int32 -> float32)

        Conversion Examples:
        -------------------
        Master: S32LE stereo [L, R]
        Reader wants: S16LE mono from left channel
        → Extract channel 0, convert int32 -> int16

        Master: S16LE stereo [L, R]
        Reader wants: F32LE mono from right channel
        → Extract channel 1, convert int16 -> float32

        Args:
            audio_data: Audio samples in master buffer format
            reader_config: Target reader configuration

        Returns:
            Audio data in reader's requested format and channels

        Raises:
            ValueError: If reader requests more channels than available in audio data

        Supported Conversions:
        --------------------
        - S32LE ↔ S16LE ↔ F32LE
        - Channel selection from multi-channel (up to 8 channels)
        - Readers can only extract channels that exist in master buffer
        """
        master_format = self._normalize_format(self.capture_config.format)
        reader_format = reader_config.format

        # ===== STEP 1: Extract channels =====
        if audio_data.ndim == 2:
            # Multi-channel master buffer
            available_channels = audio_data.shape[-1]

            # Runtime validation: ensure reader doesn't request more channels than available
            if reader_config.channels > available_channels:
                raise ValueError(
                    f"Reader requests {reader_config.channels} channels, "
                    f"but audio data only has {available_channels} channels"
                )

            # Validate channel indices
            max_requested_index = max(reader_config.channel_indices)
            if max_requested_index >= available_channels:
                raise ValueError(
                    f"Reader requests channel index {max_requested_index}, "
                    f"but audio data only has channels 0-{available_channels - 1}"
                )

            valid_indices = [min(idx, available_channels - 1) for idx in reader_config.channel_indices]

            if reader_config.channels == 1:
                # Extract single channel (mono)
                audio_data = audio_data[:, valid_indices[0]]
            else:
                # Extract multiple channels (up to 8)
                audio_data = audio_data[:, valid_indices]
        elif audio_data.ndim == 1:
            # Mono master buffer
            if reader_config.channels > 1:
                raise ValueError(
                    f"Reader requests {reader_config.channels} channels, but audio data is mono (1 channel)"
                )
            # Mono master buffer, reader wants mono - use as-is

        # ===== STEP 2: Format conversion =====
        if master_format == reader_format:
            # No format conversion needed
            return audio_data

        # Convert TO F32LE (float32)
        if reader_format == "F32LE":
            scale_map = {"S32LE": 2147483648.0, "S16LE": 32768.0}
            scale = scale_map.get(master_format, 1.0)
            converted = audio_data.astype(np.float32) / scale
            return np.nan_to_num(converted, nan=0.0)

        # Convert TO S16LE (int16)
        if reader_format == "S16LE":
            if master_format == "S32LE":
                # S32 -> S16: Drop lower 16 bits
                return (audio_data >> 16).astype(np.int16)
            elif master_format == "F32LE":
                # F32 -> S16: Scale and clip, handle NaN
                cleaned = np.nan_to_num(audio_data, nan=0.0)
                return (np.clip(cleaned, -1.0, 1.0) * 32767.0).astype(np.int16)

        # Convert TO S32LE (int32)
        if reader_format == "S32LE":
            if master_format == "S16LE":
                # S16 -> S32: Shift up 16 bits
                return audio_data.astype(np.int32) << 16
            elif master_format == "F32LE":
                # F32 -> S32: Scale and clip, handle NaN
                cleaned = np.nan_to_num(audio_data, nan=0.0)
                return (np.clip(cleaned, -1.0, 1.0) * 2147483647.0).astype(np.int32)

        return audio_data

    # ===== CAPTURE PROCESSING =====

    def _process_captured_audio(self, audio_data: np.ndarray):
        """
        Process captured audio: apply fade-in, record, and store in master buffer.

        This is called by the backend for each captured audio frame.

        Processing Steps:
        ----------------
        1. Drop first 3 buffers (hardware transient noise)
        2. Apply fade-in to first valid buffer (avoid clicks)
        3. Ensure 2D shape for multi-channel
        4. Record to WAV file if enabled
        5. Store in master buffer (circular deque)
        6. Update write_index (absolute position)
        7. Notify waiting readers via condition variable

        Args:
            audio_data: Raw audio from capture device
                       Shape: (samples,) for mono or (samples, channels) for stereo

        Thread Safety:
            Uses master_buffer_lock for atomic buffer updates
            Notifies new_data_cv to wake up blocking readers
        """
        master_format = self._normalize_format(self.capture_config.format)
        expected_dtype = self._get_dtype_for_format(master_format)

        # Ensure correct dtype
        if audio_data.dtype != expected_dtype:
            audio_data = audio_data.astype(expected_dtype)

        # Drop first few buffers (transient noise from device startup)
        if self.buffers_dropped < 3:
            self.buffers_dropped += 1
            return

        # Apply fade-in to first valid buffer (avoid audio clicks)
        if not self.fade_in_done:
            fade_factor = np.linspace(0.0, 1.0, len(audio_data), dtype=np.float32)
            if audio_data.ndim > 1:
                fade_factor = fade_factor.reshape(-1, 1)

            if audio_data.dtype == np.float32:
                audio_data = audio_data * fade_factor
            else:
                audio_data = (audio_data.astype(np.float32) * fade_factor).astype(expected_dtype)

            self.fade_in_done = True

        # Ensure 2D shape for multi-channel consistency
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)

        # Record to WAV file if enabled
        if self.audio_capture_recorder:
            self.audio_capture_recorder.write(audio_data)

        # Store in master buffer (thread-safe)
        with self.master_buffer_lock:
            # Append each sample frame to circular buffer
            for sample_frame in audio_data:
                self.master_buffer.append(sample_frame)

            # Update absolute write position
            self.write_index += len(audio_data)

            # Wake up all readers waiting for new data (event-driven)
            self.new_data_cv.notify_all()

    # ===== PLAYBACK PROCESSING =====
    def _prepare_playback_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Prepare audio for playback: normalize, clip, record, adjust channels, and convert format.

        Processing Steps:
        ----------------
        1. Record to WAV file if enabled
        2. Adjust channels (mono->stereo or stereo->mono)
        3. Convert to target playback format

        Args:
            audio_data: Audio samples (any format)

        Returns:
            Audio data in playback_config.format, ready for device output
        """

        # Record to WAV file if enabled
        if self.audio_playback_recorder and self.audio_playback_recorder.is_recording:
            self.audio_playback_recorder.write(audio_data)

        # Adjust channels to match playback configuration
        if audio_data.ndim == 1 and self.playback_config.channels == 2:
            # Mono -> Stereo: duplicate to both channels
            audio_data = np.stack([audio_data, audio_data], axis=-1)
        elif audio_data.ndim == 2 and self.playback_config.channels == 1:
            # Stereo -> Mono: average channels
            audio_data = np.mean(audio_data, axis=-1)

        # Convert to target playback format
        if self.playback_config.format == "S32LE":
            # Scale float [-1, 1] to int32 range and handle NaN values
            playback_data = (np.nan_to_num(audio_data, nan=0.0) * 2147483647.0).astype(np.int32)
        elif self.playback_config.format == "S16LE":
            # Scale float [-1, 1] to int16 range and handle NaN values
            playback_data = (np.nan_to_num(audio_data, nan=0.0) * 32767.0).astype(np.int16)
        else:  # F32LE
            # Already float, just ensure correct dtype and handle NaN
            playback_data = np.nan_to_num(audio_data, nan=0.0).astype(np.float32)

        return playback_data

    def signal_stream_end(self):
        """
        Signal that the current audio stream has ended.

        This indicates the playback loop the end of a sequence.

        Called by TTS adapter when END_TOKEN is processed.

        Thread Safety:
            Protected by playback_queue_lock
            Notifies playback_queue_cv to wake playback thread
        """
        with self.playback_queue_lock:
            self.stream_end_signaled.set()
            self.playback_queue_cv.notify_all()
        logger.debug("Stream end signaled to audio manager")

    def play_audio_async(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """
        Queue audio for playback without blocking.

        This is the main API for playing audio. Audio is queued and played
        asynchronously by the playback thread. Playback will start automatically
        if not already running.

        Args:
            audio_data: Audio samples (any format, will be converted)
            sample_rate: Sample rate for this audio (can differ per chunk)
                        If None, uses default playback_config.sample_rate

        Thread Safety:
            Protected by playback_queue_lock
            Notifies playback_queue_cv to wake playback thread

        Example:
            # Play TTS audio at 16kHz
            manager.play_audio_async(tts_audio, sample_rate=16000)

            # Play notification at 22kHz
            manager.play_audio_async(beep, sample_rate=22050)

            # Sample rates can change dynamically!
        """
        sr = sample_rate or self.playback_config.sample_rate

        # Add to queue FIRST (before starting thread)
        with self.playback_queue_lock:
            self.playback_queue.append((audio_data, sr))
            # Clear stream end signal when new audio is queued
            self.stream_end_signaled.clear()

        # Auto-start playback if not running (on-demand mode)
        if not self.playback_running.is_set():
            logger.info("Auto-starting playback (on-demand)")
            self.start_playback()
        else:
            # Thread already running, wake it up
            with self.playback_queue_lock:
                self.playback_queue_cv.notify()

    # ===== STATUS CHECKS =====

    def is_capture_running(self) -> bool:
        """
        Check if capture is active.

        Returns:
            True if capture thread is running
        """
        return self.capture_running.is_set()

    def is_playing(self) -> bool:
        """
        Check if audio is currently being written to device.

        Returns:
            True if actively pushing audio to hardware
        """
        return self.playback_active.is_set()

    def is_playback_queue_empty(self) -> bool:
        """
        Check if playback queue is empty.

        Returns:
            True if no audio queued for playback

        Thread Safety:
            Protected by playback_queue_lock
        """
        with self.playback_queue_lock:
            return len(self.playback_queue) == 0

    def is_playback_complete(self) -> bool:
        """
        Check if playback is completely finished.

        Playback is complete when:
        1. Queue is empty (no more audio to play)
        2. No audio is currently being written to device

        Returns:
            True if all playback is done

        Thread Safety:
            Uses locks for both queue and active flag checks

        Example:
            # Wait for all audio to finish
            while not manager.is_playback_complete():
                time.sleep(0.1)
        """
        with self.playback_queue_lock:
            queue_empty = len(self.playback_queue) == 0
        return queue_empty and not self.playback_active.is_set()

    # ===== SHUTDOWN =====

    def shutdown(self):
        """
        Stop all operations and clean up resources.

        Shutdown Steps:
        --------------
        1. Stop capture (closes device, stops thread)
        2. Stop playback (closes device, stops thread)
        3. Stop any active recordings
        4. Disable and clear all readers
        5. Clear master buffer

        This should be called before program exit to ensure clean shutdown.

        Thread Safety:
            Uses appropriate locks for all cleanup operations

        Example:
            try:
                # Use audio manager
                manager.start_capture()
                # ... do work ...
            finally:
                manager.shutdown()  # Always cleanup
        """
        logger.info(f"Shutting down {self.__class__.__name__}")

        # Stop audio I/O
        self.stop_capture()
        self.stop_playback()

        # Stop playback recording
        if self.audio_playback_recorder:
            self.audio_playback_recorder.stop()

        # Cleanup all readers
        with self.readers_lock:
            for reader in self.readers.values():
                reader.disable()
            self.readers.clear()

        # Clear master buffer
        with self.master_buffer_lock:
            self.master_buffer.clear()
