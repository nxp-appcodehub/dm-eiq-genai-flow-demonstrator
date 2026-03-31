# Chat Interface GUI Module

A PyQt6-based graphical user interface example for the eIQ GenAI Flow conversational AI pipeline.

## Overview

The Chat Interface provides a **mock GUI interface** with real-time conversation bubbles, status indicators, and seamless integration with the eIQ GenAI Flow pipeline through message queues — useful for demos or extending into a custom UI.

## Features

- **Real-time Chat Bubbles**: User questions and AI responses displayed in conversation format
- **Status Indicators**: Visual feedback for connection, listening, and wake-word states
- **Streaming Responses**: AI responses appear word-by-word as they generate

## Usage

### Method 1: Launch Script (Recommended)

```bash
./launch_gui.sh
```
This script:

* Initializes the **IPC queues**
* Checks if the **dependencies** are installed
* **Cleans** existing processes
* **Runs eIQ GenAI Flow** with configuration defined in the script
* **Runs the GUI**

The GUI waits for eIQ GenAI Flow to be ready, then displays messages exchanged in real time.

You can freely **modify or extend** this script to integrate your own GUI or automation logic.

Note: some path and pipeline configuration may need to be customized

### Method 2: Direct Python Execution

```bash
# Start the GUI
python3 src/chat_interface/chat_interface.py &

# Start eIQ GenAI Flow with chat interface mode
cd ../../../
python3 eiq_genai_flow.py -i chat_interface -o tts -r
```

## Configuration

### Launch Script Parameters

Edit `launch_gui.sh` to customize:

```bash
EGF_MODEL="-m danube-500M-q8"                       # LLM model
EGF_ASR_TYPE="-a moonshine-base"                    # ASR model
EGF_USE_RAG=true                                    # Enable RAG
EGF_USE_NEUTRON=true                                # Enable NPU acceleration
EGF_CAPTURE_DEVICE=""                               # Audio input device
EGF_PLAYBACK_DEVICE=""                              # Audio output device
EGF_OUTPUT_MODE="-o tts"                            # Text-to-speech output mode
VIT_MODEL="-w vit/models/VIT_Model_en.bin"          # VIT wake-word Model
```

### Audio Device Configuration

```bash
# Auto-detect devices (default)
EGF_CAPTURE_DEVICE=""
EGF_PLAYBACK_DEVICE=""

# Specify devices manually
EGF_CAPTURE_DEVICE="plughw:CARD=Talk"
EGF_PLAYBACK_DEVICE="plughw:CARD=Talk"
```

## Status Indicators


| Symbol                   | Meaning                                                      |
|--------------------------| -------------------------------------------------------------- |
| `○ Connecting...`        | Establishing connection to eIQ GenAI Flow                    |
| `● Connected!`           | Successfully connected and ready                             |
| `◉ Listening...`         | Actively recording audio input                               |
| `◎ Say the wake-word...` | Waiting for the wake-word as defined in the VIT model passed |
| `▶ `                     | Command or intent recognized                                 |
| `○ Disconnected`         | Connection lost or terminated                                |

## Architecture

### Message Queue Communication

The GUI communicates with eIQ GenAI Flow through POSIX message queues:

- `/dev/mqueue/egf_to_gui` - Receives messages from eIQ GenAI Flow
- `/dev/mqueue/gui_to_egf` - Sends messages to eIQ GenAI Flow

### Message Protocol

The GUI listens to messages from the eIQ GenAI Flow system using a queue-based protocol.
Each message is formated as: `prefix:message`. The prefix indicates the type of event or data being sent.

**Incoming Messages:**

- `CON:` - Connection established
- `QST:<text>` - User question text
- `RSP:<text>` - AI response text
- `WWD:` - Wake-word detected
- `VIS:` - VIT started
- `CMD:<text>` - System message detected
- `DIS:` - Disconnected

**Special Tokens:**

- `<end>` - End of message
- `<stop>` - Stop/cancel operation

## Architecture Overview

```
┌────────────────────────────────┐                     ┌───────────────────────┐
│         eIQ GenAI Flow         │ ─────────────────▷  │   Chat Interface UI   │
│ (VIT, ASR, RAG, LLM, VLM, TTS) │   posix_ipc Queue   │  (PyQt6 mock client)  │
└────────────────────────────────┘                     └───────────────────────┘
```

The Chat Interface acts as a **client** that visually renders messages exchanged over an **IPC queue**, simulating an interactive voice assistant or conversational system.

## Troubleshooting

### GUI Won't Start

```bash
# Check PyQt6 installation
python3 -c "import PyQt6; print('PyQt6 OK')"

# Check posix_ipc installation
python3 -c "import posix_ipc; print('posix_ipc OK')"

# Reinstall if needed
pip install -e . --force-reinstall
```

### Connection Issues

```bash
# Check message queues
ls -la /dev/mqueue/

# Clean up queues
./launch_gui.sh -k

# Restart both processes
./launch_gui.sh
```

### Audio Issues

```bash
# List available audio devices
arecord -l  # Input devices
aplay -l    # Output devices

# Test audio
arecord -D plughw:CARD=CAPTURE_DEVICE  -d 10 -r 16000 -c 2 -f FLOAT_LE > out.pcm
aplay -D plughw:CARD=PLAYBACK_DEVICE  -d 10 -r 16000 -c 2 -f FLOAT_LE out.pcm
```

*Note: change "CAPTURE_DEVICE" and "PLAYBACK_DEVICE" by the devices names returned by aplay/arecord -l*

## License

Copyright 2025 NXP. This software is proprietary and subject to [LA_OPT_Online Code Hosting NXP_Software_License - v1.4 May 2025](LICENSE.txt)
