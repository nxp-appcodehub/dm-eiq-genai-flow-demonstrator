# Audio Manager Module

Unified Audio Manager for capture and playback with multiple backend support (ALSA, GStreamer).

## Features

- Multi-reader audio capture with independent read pointers
- Configurable audio formats and sample rates
- Support for both ALSA and GStreamer backends
- Audio recording to WAV files
- Thread-safe buffer management
- Playback queue management
- **Command-line interface for testing and demos**

## Installation

### Basic installation
```bash
pip install .
```

### With ALSA backend
```bash
pip install .[alsa]
```

### With GStreamer backend
```bash
pip install .[gstreamer]
```

### With all backends
```bash
pip install .[all]
```

### Development installation
```bash
pip install -e .[dev]
```

## Command-Line Usage

The module can be run directly from the command line:

### Show backend information
```bash
python -m audio_manager info
```

### Test capture and playback
```bash
python -m audio_manager test --duration 3
```

### Record audio to file
```bash
python -m audio_manager record --duration 5 --output-dir ./recordings
```

### Play test tones
```bash
python -m audio_manager play-tone
```

### Demonstrate multiple readers
```bash
python -m audio_manager multi-reader --duration 2
```

### Advanced options
```bash
# Use specific backend
python -m audio_manager test --backend alsa

# Use specific devices
python -m audio_manager test --capture-device hw:0,0 --playback-device hw:1,0

# Custom sample rate and channels
python -m audio_manager test --sample-rate 48000 --channels 1

# Get help
python -m audio_manager --help
python -m audio_manager test --help
```

During initialization, the audio manager will automatically apply
device-specific configurations by calling `set_capture_device_config` and
`set_playback_device_config` from `audio_manager/set_audio_device_config.py`.
These functions set hardware-specific parameters like volume levels, quality settings,
and other codec-specific configurations based on the detected audio device.


## Python API Usage

```python
from audio_manager import create_audio_manager, CaptureConfig, PlaybackConfig

# Create configurations
capture_config = CaptureConfig(
    capture_device="default",
    sample_rate=16000,
    channels=2,
    format="S32LE",
)

playback_config = PlaybackConfig(
    playback_device="default",
    sample_rate=16000,
    channels=1,
    format="S32LE",
)

# Create audio manager (auto-selects backend)
audio_manager = create_audio_manager(
    backend="auto",  # or "alsa", "gstreamer"
    capture_config=capture_config,
    playback_config=playback_config,
)

# Start capture and playback
audio_manager.start_capture()
audio_manager.start_playback()
```

## Available Commands

| Command | Description |
|---------|-------------|
| `info` | Display available backends and their status |
| `test` | Test audio capture and playback (record then play) |
| `record` | Record audio to WAV file |
| `play-tone` | Play a musical scale (test playback) |
| `multi-reader` | Demonstrate multiple readers with different configs |

## License

Copyright 2026 NXP - Confidential and Proprietary
```

## 4. Usage examples

Now users can run the module directly:

```bash
# Check what backends are available
python -m audio_manager info
```

```bash
# Quick test (3 seconds record + playback)
python -m audio_manager test
```

```bash
# Record 10 seconds to file
python -m audio_manager record --duration 10
```

```bash
# Test playback with musical tones
python -m audio_manager play-tone
```

```bash
# Test with specific backend
python -m audio_manager test --backend alsa --duration 5
