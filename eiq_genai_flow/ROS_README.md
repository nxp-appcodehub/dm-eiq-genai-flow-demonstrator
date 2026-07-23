# ROS 2 Integration for eIQ® GenAI Flow

This guide explains how to integrate and use the eIQ GenAI Flow with ROS 2, enabling robotic applications to leverage conversational AI capabilities.

---

## Overview

The ROS 2 wrapper for eIQ GenAI Flow provides a node-based interface to the conversational AI pipeline, allowing seamless integration with ROS 2 ecosystems. It exposes the pipeline's functionality through standard ROS 2 topics and services.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [ROS 2 Interface](#ros-2-interface)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

<a name="architecture"></a>
## Architecture

The ROS 2 node wraps the eIQ GenAI Flow pipeline and provides:

**Publishers** (Output Events):
- `/genai/wakeword` - Wake-word detection events
- `/genai/vad/event` - Voice Activity Detection events
- `/genai/voice_id/event` - All Voice ID events
- `/genai/stt/transcription` - Speech-to-text transcriptions
- `/genai/rag/category` - RAG query classification
- `/genai/rag/answer` - RAG retrieved answers
- `/genai/llm/token` - LLM token stream
- `/genai/llm/response` - Complete LLM responses
- `/genai/tts/event` - TTS synthesis and playback events

**Services**:
- `/genai/trigger_listening` - Trigger STT listening (replaces Enter key)

**Subscribers** (Input):
- `/genai/text_input` - Send text queries to the system

---

<a name="prerequisites"></a>
## Prerequisites

### System Requirements

- **ROS 2 Distribution**: Jazzy Jalisco
- **Platform**: i.MX95 or 8MP
- **Robotics Edge Platform BSP**: NXP Linux BSP (see [Documentation](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/robotics-edge-platform:ROBOTICS-EDGE-PLATFORM))
- **eIQ GenAI Flow**: Installed and configured (see [Installation](#installation))

---

<a name="installation"></a>
## Installation

### Install eIQ GenAI Flow with ROS 2 Support

First, ensure the eIQ GenAI Flow demonstrator is installed on your device:

```bash
# Transfer the eiq_genai_flow folder to your i.MX device
scp -r eiq_genai_flow root@<imx-device-ip>:/root/

# SSH into the device
ssh root@<imx-device-ip>

# Install dependencies
cd /root/eiq_genai_flow
source ./install_ROS.sh
```

---

<a name="ros-2-interface"></a>
## ROS 2 Interface

### Topics

#### Publishers (Output)

| Topic | Type | Description |
|-------|------|-------------|
| `/genai/wakeword` | `std_msgs/String` | Wake-word detection with energy level |
| `/genai/vad/event` | `std_msgs/String` | Speech start/end events |
| `/genai/voice_id/event` | `std_msgs/String` | Voice identification events |
| `/genai/stt/transcription` | `std_msgs/String` | Final transcribed text |
| `/genai/rag/category` | `std_msgs/String` | Query category (ACCEPTED, REJECTED, etc.) |
| `/genai/rag/answer` | `std_msgs/String` | Retrieved answer from knowledge base |
| `/genai/llm/token` | `std_msgs/String` | Individual LLM tokens (streaming) |
| `/genai/llm/response` | `std_msgs/String` | Complete LLM response |
| `/genai/tts/event` | `std_msgs/String` | TTS synthesis and playback events |

#### Subscribers (Input)

| Topic | Type | Description |
|-------|------|-------------|
| `/genai/text_input` | `std_msgs/String` | Send text queries to the system |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/genai/trigger_listening` | `std_srvs/Trigger` | Start STT listening (for KASR mode) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_mode` | string | `vasr` | Input mode: `vasr`, `kasr`, `keyb` |
| `capture_device` | string | `plughw:wm8962audio` | Audio capture device |
| `wake_word_model` | string | `VIT_Model_en.bin` | Path to VIT wake-word model |
| `llm_model` | string | `danube-500M-q8` | LLM model name or `no_llm` |
| `output_mode` | string | `tts` | Output mode: `tts` or `text` |
| `playback_device` | string | `plughw:wm8962audio` | Audio playback device |
| `stt_model` | string | `moonshine-base` | STT model name |
| `stt_language` | string | `English` | STT language |
| `stt_task` | string | `transcribe` | STT task |
| `system_prompt` | string | `Helpful assistant.` | LLM system prompt |
| `use_rag` | bool | `true` | Enable RAG |
| `use_neutron` | bool | `false` | Enable NPU acceleration (i.MX95 only) |
| `use_voice_id` | bool | `false` | Enable voice identification |
| `continuous` | bool | `false` | Continuous conversation mode |
| `verbose` | bool | `true` | Verbose logging |

---

<a name="usage-examples"></a>
## Usage Examples

### Basic Usage

#### 1. Voice-Activated Mode (VASR)

Full voice interaction with wake-word detection:

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p input_mode:=vasr \
  -p llm_model:=danube-500M-q8 \
  -p use_rag:=true \
  -p verbose:=true
```

**Workflow:**
1. Say "Hey NXP" (wake-word)
2. Speak your question
3. Receive spoken response

#### 2. Keyboard-Triggered ASR (KASR)

Press Enter to start listening:

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p input_mode:=kasr \
  -p llm_model:=danube-500M-q8
```

**Trigger listening via service:**
```bash
# In another terminal
ros2 service call /genai/trigger_listening std_srvs/srv/Trigger
```
#### 3. Keyboard Text Input (KEYB)

Text-only interaction:

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p input_mode:=keyb \
  -p output_mode:=text \
  -p llm_model:=danube-500M-q8
```

**Send text query:**
```bash
ros2 topic pub --once /genai/text_input std_msgs/msg/String \
  "{data: 'What is diabetes?'}"
```

#### 4. Voice Identification (Voice ID)

Enable speaker identification and verification:

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p input_mode:=vasr \
  -p use_voice_id:=true \
  -p llm_model:=danube-500M-q8
```

**Monitor Voice ID events:**
```bash
ros2 topic echo /genai/voice_id/event
```

**Workflow:**
1. Say "Hey NXP" (wake-word)
2. System identifies the speaker
3. Speak your question
4. Receive personalized response
5. Then the speaker is remembered for future interactions (no need to use wake-word again for the same speaker). If you want to change the speaker, say "Hey NXP" again.


### Advanced Configuration

#### Custom Audio Devices

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p capture_device:=plughw:CARD=wm8960audio \
  -p playback_device:=plughw:CARD=wm8960audio \
  -p input_mode:=vasr
```

#### NPU Acceleration (i.MX95 B0)

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p use_neutron:=true \
  -p llm_model:=danube-500M-q8
```

#### Continuous Conversation Mode

```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p continuous:=true \
  -p input_mode:=vasr
```

### Monitoring Events

#### Monitor All Topics

```bash
# Terminal 1: Run the node
ros2 run imx_genai eiq_genai_flow

# Terminal 2: Monitor wake-word detection
ros2 topic echo /genai/wakeword

# Terminal 3: Monitor wake-word detection
ros2 topic echo /genai/vad/event

# Terminal 4: Monitor wake-word detection
ros2 topic echo /genai/voice_id/event

# Terminal 5: Monitor STT transcriptions
ros2 topic echo /genai/stt/transcription

# Terminal 6: Monitor LLM responses
ros2 topic echo /genai/llm/response

# Terminal 7: Monitor TTS events
ros2 topic echo /genai/tts/event
```

#### Monitor RAG Events

```bash
# Monitor RAG category
ros2 topic echo /genai/rag/category

# Monitor RAG answers
ros2 topic echo /genai/rag/answer
```

#### Monitor LLM Token Stream

```bash
# Watch tokens as they're generated
ros2 topic echo /genai/llm/token
```

---

<a name="configuration"></a>
## Configuration

### Launch Files

Create a launch file for easier configuration:

```python
# ~/ros2_ws/src/imx_genai/launch/genai_flow.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='imx_genai',
            executable='eiq_genai_flow',
            name='eiq_genai_flow_node',
            output='screen',
            parameters=[{
                'input_mode': 'vasr',
                'capture_device': 'plughw:wm8962audio',
                'playback_device': 'plughw:wm8962audio',
                'llm_model': 'danube-500M-q8',
                'stt_model': 'moonshine-base',
                'use_rag': True,
                'use_neutron': False,
                'use_voice_id': False,
                'continuous': False,
                'verbose': True,
            }]
        )
    ])
```

**Usage:**
```bash
ros2 launch imx_genai genai_flow.launch.py
```

### Parameter Files

Create a YAML parameter file:

```yaml
# ~/ros2_ws/src/imx_genai/config/genai_params.yaml
eiq_genai_flow_node:
  ros__parameters:
    input_mode: "vasr"
    capture_device: "plughw:wm8962audio"
    playback_device: "plughw:wm8962audio"
    wake_word_model: "/root/eiq_genai_flow/vit/src/vit/models/VIT_Model_en.bin"
    llm_model: "danube-500M-q8"
    output_mode: "tts"
    stt_model: "moonshine-base"
    stt_language: "English"
    stt_task: "transcribe"
    system_prompt: "You are a helpful robotic assistant."
    use_rag: true
    use_neutron: false
    use_voice_id: false
    continuous: false
    verbose: true
```

**Usage:**
```bash
ros2 run imx_genai eiq_genai_flow --ros-args \
  --params-file ~/ros2_ws/src/imx_genai/config/genai_params.yaml
```

---

<a name="troubleshooting"></a>
## Troubleshooting

### Common Issues

#### 1. Module Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'eiq_genai_flow'
```

**Solution:**
```bash
source /root/eiq_gennai_flow/venv/bin/activate
```

#### 2. Audio Device Not Found

**Error:**
```
ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM plughw:wm8962audio
```

**Solution:**
```bash
# List available devices
arecord -l
aplay -l

# Update parameters with correct device
ros2 run imx_genai eiq_genai_flow --ros-args \
  -p capture_device:=plughw:CARD=YourDevice,DEV=0
```

#### 3. SD Card Full / No Space Left on Device

**Error:**
```
No space left on device
```

**Solution:**

After flashing the BSP image, the root filesystem may not use the full SD card capacity. Resize it with:
```bash
resize2fs /dev/mmcblk1p2
```

#### 4. SSL Certificate Verification Errors

**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```
or
```
pip: SSL error when downloading packages
```

**Solution:**

SSL certificates are validated against the current date. If the board clock is wrong, certificates will appear expired:
```bash
# Check current date/time
date

# Set date manually (example)
date -s "2026-07-30 17:00:00"

# Or sync via NTP if network is available
timedatectl set-ntp true
```

#### 5. ROS 2 Node Not Starting

**Check:**
```bash
# Verify ROS 2 environment
echo $ROS_DISTRO

# Re-source workspace
source /opt/ros/jazzy/setup.bash
source /root/eiq_genai_flow/ros2_ws/install/setup.bash

# Rebuild package
cd /root/eiq_genai_flow/ros2_ws
colcon build --packages-select imx_genai --symlink-install
```

#### 6. No Response from Services

**Debug:**
```bash
# Check if service is available
ros2 service list | grep genai

# Check service type
ros2 service type /genai/trigger_listening

# Call with verbose output
ros2 service call /genai/trigger_listening std_srvs/srv/Trigger --verbose
```

#### 7. Topics Not Publishing

**Debug:**
```bash
# List all topics
ros2 topic list

# Check topic info
ros2 topic info /genai/llm/response

# Monitor with verbose output
ros2 topic echo /genai/llm/response --verbose
```

---

## Additional Resources

- **Main Documentation**: [README.md](README.md)
- **eIQ GenAI Flow Page**: [NXP GenAI Flow](https://www.nxp.com/applications/technologies/human-machine-interface/voice-processing/simplified-and-optimized-generative-ai-at-the-edge-with-eiq-genai-flow:GEN-AI-FLOW)
- **(ROS 2 Documentation)**: [ROS 2 Official Docs](https://docs.ros.org/)
- **NXP Community**: [Generative AI & LLMs Forum](https://community.nxp.com/t5/Generative-AI-LLMs/bd-p/Generative-AI-LLMs)

---

## Support

For technical questions and support:
- **ROS 2 Integration**: Use the [NXP Community Forum](https://community.nxp.com/t5/Generative-AI-LLMs/bd-p/Generative-AI-LLMs)
- **General Issues**: See [Troubleshooting](#troubleshooting) section

---

**License**: Proprietary - NXP  
**Version**: 3.1 
**Last Updated**: July 2026