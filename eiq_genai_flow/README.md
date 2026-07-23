# eIQ® GenAI Flow

[![License badge](https://img.shields.io/badge/License-Proprietary-red)](./LICENSE)
[![Board badge](https://img.shields.io/badge/Board-i.MX95-blue)](https://www.nxp.com/products/i.MX95)
[![Board badge](https://img.shields.io/badge/Board-I.MX943-blue)](https://www.nxp.com/products/i.MX94)
[![Board badge](https://img.shields.io/badge/Board-i.MX93-blue)](https://www.nxp.com/products/i.MX93)
[![Board badge](https://img.shields.io/badge/Board-i.MX91-blue)](https://www.nxp.com/products/i.MX91)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MPLUS-blue)](https://www.nxp.com/products/I.MX8MPLUS)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MMINI-blue)](https://www.nxp.com/products/I.MX8MMINI)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MNANO-blue)](https://www.nxp.com/products/I.MX8MNANO)

[![Language badge](https://img.shields.io/badge/Language-Python-yellow)]()
[![Category badge](https://img.shields.io/badge/Category-AI/ML-green)]()

**eIQ GenAI Flow** is a software pipeline for AI-powered experiences on edge devices. The Flow supports **conversational AI** on the **NXP [i.MX9](https://www.nxp.com/products/iMX9-PROCESSORS) and [i.MX8M](https://www.nxp.com/products/i.MX8M)** applications processors.

---

## Overview

The eIQ GenAI Flow integrates multiple AI technologies to create a seamless Human-Machine Interface (HMI) experience on all kinds of edge devices. The conversational AI flow consists of the following stages, each with pre-defined, optimized models and components for end-users, as well as the ability to customize or bring their own models for each stage:

1. **Wake-Word Detection**: A VIT (Voice Intelligent Technology) Wake-Word triggers the STT (Speech-To-Text).
2. **Voice ID**: Recognize speaker identity to trigger the STT.
3. **Voice Activity Detection (VAD)**: Detects speech boundaries to determine when the user starts and stops speaking.
4. **Speech-to-Text (STT)**: Converts spoken input into text.
5. **Retrieval-Augmented Generation (RAG)**: Enhances the Large Language Model (LLM) with relevant external knowledge.
6. **Text Generation (LLM)**: Generates a response based on the retrieved context.
7. **Text-to-Speech (TTS)**: Converts the response into speech output.

![Pipeline Diagram](assets/eiq_flow.png)

For more details, use the [NXP Community Forum Generative AI & LLMs](https://community.nxp.com/t5/Generative-AI-LLMs/bd-p/Generative-AI-LLMs).

---

## Table of Contents

- [Supported NXP platforms and recommended configuration for each target](#flow-configuration-recommendations)
- [Demonstrator limitations](#limitations)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Software Components](#software-components)
  - [Voice Intelligent Technology (VIT)](#voice-intelligent-technology-vit)
  - [Voice Identification](#voice-identification)
  - [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
  - [Speech-To-Text (STT)](#speech-to-text-stt)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Large Language Model (LLM)](#large-language-model-llm)
    - [LLM on CPU/NPU (Neutron)](#llm-on-cpunpu-neutron)
    - [LLM on Discrete NPU (ARA2)](#llm-on-discrete-npu-ara2)
  - [Text-To-Speech (TTS)](#text-to-speech-tts)
  - [Audio Manager](#audio-manager)
  - [Adapters](#adapters)
- [Using NPU Acceleration](#using-npu-acceleration)
- [Benchmark mode](#benchmark-mode)
- [Audio setup](#audio-setup)
- [GUI](#graphical-user-interface-gui)
- [ROS 2 Node](#ros-node)
- [Other customizations](#other-customizations)
- [Troubleshooting](#troubleshooting)
- [Support](#support)
- [Release Notes](#release-notes)

## Supported NXP platforms and recommended configuration for each target

<a name="flow-configuration-recommendations"></a>

**eIQ GenAI Flow** can run on various i.MX platforms with different performance tiers. The following table provides easy to understand configuration recommendations that map directly to the target SoC:


| Performance Tier     | Hardware Requirements         | i.MX SOC                  | Flow Configuration | STT Models                     | LLM Models                                             | Additional Notes                           |
| -------------------- | ----------------------------- | ------------------------- | ------------------ | ------------------------------ | ------------------------------------------------------ | ------------------------------------------ |
| **High Performance** | 6+ cores, 1.8+ GHz, 8+ GB RAM | i.MX95                    | Full Flow          | whisper-small, moonshine-base  | danube-500M-q8, danube-500M-q4                         | Complete pipeline with optimal performance |
| **Standard**         | 4+ cores, 1.5+ GHz, 3+ GB RAM | i.MX952, i.MX943, i.MX8MP | Full Flow          | moonshine-base                 | danube-500M-q8 (i.MX8M/i.MX9), danube-500M-q4* (i.MX9) | Balanced performance and features          |
| **Lightweight**      | 2+ cores, 1.5+ GHz, 2+ GB RAM | i.MX93                    | Partial Flow       | moonshine-base, moonshine-tiny | danube-500M-q4                                         | LLM enabled with smaller models            |
| **Minimal**          | 2+ cores, 1.2+ GHz, 2+ GB RAM | i.MX8MN, i.MX8MM          | Retrieval Only     | moonshine-base, moonshine-tiny | None                                                   | No LLM processing                          |
| **Ultra-Light**      | 1 core, >1.2 GHz, 2+ GB RAM   | i.MX91                    | Retrieval Only     | moonshine-tiny                 | None                                                   | No LLM, no TTS                             |

**q8/q4 refers to int8 and int4 model quantization. q4 models have reduced performance on i.MX8Mx platforms with Cortex-A53 cores compared to i.MX9x Cortex-A55 architectures.*


### Configuration Details

- **Full Flow**: VIT + VoiceID + STT + RAG + LLM + TTS
- **Partial Flow**: VIT + STT + RAG + LLM + TTS (reduced model size)
- **Retrieval Only**: VIT + STT + RAG + TTS (knowledge base queries without LLM generation, except no TTS on ultra-light tier)

See [eIQ® GenAI Flow](https://www.nxp.com/applications/technologies/human-machine-interface/voice-processing/simplified-and-optimized-generative-ai-at-the-edge-with-eiq-genai-flow:GEN-AI-FLOW) for additional details and benchmarks.

---

<a name="limitations"></a>

## Demonstrator Limitations

This eIQ GenAI Flow demonstrator has the following limitations:

- **Session timeout**: The demonstrator automatically shuts down after 1 hour of operation. The timeout can be extended or removed for production usage on request
- **Language support**: The demonstrator supports English language only. The complete and partial GenAI Flow can be extended to support other languages beyond the English-only demonstrator (🇨🇳, 🇪🇸, 🇩🇪, 🇰🇷, 🇯🇵, 🇫🇷, etc...). If interested in production-use of GenAI flow with additional languages (for STT-only or with the complete Flow) and for additional languages/accents (for TTS / speech generation), please use our community forum or make the request to your NXP account or regional manager
- **Component delivery**: All components are delivered as pre-built Python wheels and can be imported in other i.MX8M/9 projects
- **Model selection**: Includes a curated subset of STT and LLM models optimized for the target platforms
- **Model format**: Models are delivered in an encrypted format

These limitations are designed to provide an optimal evaluation experience while showcasing the capabilities of the eIQ GenAI Flow on NXP platforms.

---

<a name="installation"></a>


## Installation of the demonstrator package

### BSP selection

This demonstrator requires a Linux BSP available at [Embedded Linux for i.MX Applications Processors](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX).
The NPU Acceleration is available only for **i.MX95 B0** devices with the following requirements:

- **BSP Version**: L6.12.49-2.2.0 or later ([Download L6.12.49-2.2.0_MX95](https://www.nxp.com/webapp/sps/download/license.jsp?colCode=L6.12.49-2.2.0_MX95&appType=file2&DOWNLOAD_ID=null))
- **Device Tree Configuration**: Extended CMA memory region (>= 3GB) - see [CMA Configuration](#cma-configuration)

The NPU Acceleration provides:

- Significant CPU load reduction
- Faster Time-To-First-Token (TTFT) for LLM operations
  The demonstrator will run on CPUs only on all other platforms.

### Get the demonstrator package

It's recommended to get the package on a PC host, then copy it to the i.MX device. The following section describes how to set up the Linux PC host, clone the repository, and prepare the demonstrator for deployment on i.MX devices.

#### Install Git LFS on the Linux PC host

This repository uses [Git Large File Storage (LFS)](https://git-lfs.github.com/) to manage large files (e.g., models, datasets, binaries).

**Before cloning this repository**, ensure that Git LFS is installed and initialized on your host machine.

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install git-lfs
```

#### Initialize Git LFS on the Linux PC host (Run Once)

```bash
git lfs install
```

#### Cloning the project on the Linux PC host

```bash
git clone --single-branch -b release/v3.1 https://github.com/nxp-appcodehub/dm-eiq-genai-flow-demonstrator
cd dm-eiq-genai-flow-demonstrator
```

Git LFS will automatically download all tracked large files during or after the clone. If needed, you can run:

```bash
git lfs pull
```

to manually fetch any missing LFS files.

Once fully cloned, copy the **eiq_genai_flow** folder from this package to the target i.MX device running the NXP BSP (e.g., to the SD card).

**Requirements:**

- At least 16GB of free space on the target device
- SSH or physical access to transfer files

**Example transfer command:**

```bash
scp -r eiq_genai_flow root@<imx-device-ip>:/root/
```

---

<a name="getting-started"></a>

## Getting Started

### Install dependencies on the i.MX target

After transferring the **eiq_genai_flow** folder to your i.MX device, install the required dependencies before running the demonstrator for the first time, with the `install.sh` script

<a name="install-script"></a>

**What this does:**

- Synchronizes system time via NTP (unless `--skip-date` is used)
- Installs ALSA development headers and espeak-ng (built from source when not already present)
- Installs a CPU-only build of `torch` (and `torchaudio` when pinned) from the PyTorch CPU index
- Installs all required Python dependencies, including the eIQ GenAI Flow pipeline modules that are delivered as pre-built wheels (see [How the pipeline modules are installed](#module-wheels))

**Options:**


| Option               | Description                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `--dev`              | Install with the development dependencies (`tenacity`, `pandas`, `altgraph`)               |
| `--gui`              | Install the optional chat interface GUI (pulls in PySide6)                                 |
| `--skip-date`        | Skip the NTP system time synchronization                                                   |
| `--venv`             | Install into a virtual environment (`./venv`) instead of system-wide                       |
| `--python VERSION`   | Select the Python interpreter, e.g.`--python 3.13` or `--python 3.14` (default: `python3`) |
| `--recreate-venv`    | Remove and recreate the virtual environment (only with`--venv`)                            |
| `--no-auto-activate` | Do not add venv auto-activation to the shell startup file (only with`--venv`)              |
| `--no-fix-record`    | Skip fixing system packages with missing `RECORD` files (fixed automatically by default)    |
| `-h`, `--help`       | Show the help message and exit                                                             |

**Common examples:**

```bash
./install.sh                            # Default system-wide installation
./install.sh --skip-date                # Skip time synchronization
./install.sh --dev                      # Include development dependencies
./install.sh --gui                      # Include the optional chat interface GUI
./install.sh --python 3.14              # Use a specific Python version
./install.sh --venv                     # Use a virtual environment instead of system-wide
./install.sh --venv --recreate-venv     # Recreate the virtual environment
./install.sh --venv --no-auto-activate  # Skip venv auto-activation in shell
./install.sh --gui --venv --python 3.14 # Combine options
./install.sh --no-fix-record            # Skip the RECORD-file fixup step
./install.sh --help                     # Show all options
```

**Installation modes:**

- **System-wide (default)**: Packages are installed globally using `sudo pip3`. Available to all users and sessions.

  ```bash
  ./install.sh
  ```
- **Virtual environment**: Isolated Python environment at `./venv` with system site packages enabled.

  ```bash
  ./install.sh --venv
  ```

  When using `--venv`, the virtual environment is automatically activated in your shell startup file (~/.bashrc), unless `--no-auto-activate` is passed.

<a name="module-wheels"></a>

### How the pipeline modules are installed

The eIQ GenAI Flow pipeline modules are **not** built from source at install time. They are distributed as pre-built Python **wheels** and are declared as dependencies of the `eiq_genai_flow` package in `pyproject.toml`. The following modules are pulled in as wheels:

- `nxp_eiq_shared_utils` - shared utilities
- `nxp_eiq_speech_to_text` - Speech-To-Text (STT)
- `nxp_eiq_text_to_speech` - Text-To-Speech (TTS)
- `nxp_eiq_vad` - Voice Activity Detection (VAD)
- `nxp_eiq_voice_id` - Voice Identification
- `nxp_eiq_vit_wake_word` - VIT wake-word engine
- `nxp_eiq_llm` - Large Language Model runtime
- `nxp_eiq_retriever` - Retrieval-Augmented Generation (RAG)
- `nxp_eiq_audio_manager` - Audio Manager
- `nxp_eiq_chat_interface` - optional chat interface GUI (installed only with `--gui`)


The correct wheel is selected automatically for the running platform and Python version:

- **aarch64 (i.MX targets)**: cythonized, architecture-specific wheels (`...-cpXX-cpXX-linux_aarch64.whl`).
- Pure-Python modules ship as `...-pyXX-none-any.whl` and are shared across architectures.

By default the wheels are downloaded from the NXP Nexus repository. When a local `wheels/` folder is present next to `install.sh` (as produced for offline release packages), `install.sh` automatically resolves the module wheels from that folder instead, making the package self-contained and installable without network access to Nexus.



### Running the demonstrator

Once the dependencies are installed, to run the demonstrator, use the following basic command to run default configuration:

```bash
eiq_genai_flow
```


> Run ```eiq_genai_flow --help``` to see available options.

The default mode is keyboard-to-speech, meaning the module VIT and STT are disabled. To enable the speech-to-speech experience use the  `--input-mode vasr` argument.

The application supports various input/output options and model selections, which are detailed in the software components sections below.

---

<a name="software-components"></a>

## Software Components

<a name="voice-intelligent-technology-vit"></a>

### Voice Intelligent Technology (VIT)

VIT is NXP’s Voice UI technology that enables always-on Wake-Word detection using deep learning.

VIT is integrated with **"HEY NXP"** pre-defined Wake-Word.

**✅ Enabling VIT**

Use the `-i vasr` argument to enable STT **after the Wake-Word detection**.

Additional options include:

- `-c` (continuous mode): Allows continuous conversation without requiring the Wake-Word after each response.
- `--voice-id` (Voice ID mode): Enables speaker recognition so the pipeline identifies the speaker before processing the request. Use it together with `-i vasr` (e.g. `eiq_genai_flow -i vasr --voice-id`).

#### Custom Wake-Word Models

You can use custom wake-word models with the `-w/--wake-word-model` option:

```bash
eiq_genai_flow -i vasr -w path/to/your/custom_model.bin
```

**Creating Custom Wake-Word Models:**

1. **Generate the model** at the [VIT Model Generation Tool](https://vit.nxp.com/#/)

   - Click on LOGIN and create an NXP account or sign in
   - Click on GENERATE MODEL
   - In the VOICE COMMAND section, Click on CHOOSE
   - Select **"Linux BSP"** for SW platform & version
   - Choose **"VIT 4.13.0"** for VIT version
   - Select any device (your choice)
   - Define your custom wake words (up to 3)
   - Generate and download the model bin file
2. **Use your custom model:**

   ```bash
   eiq_genai_flow -i vasr -w VIT_Model_yourmodel.bin
   ```

**Note:**: The default VIT wake-word model to use can be defined in the config.py.

⎺⎺⎺

<a name="voice-identification"></a>

### Voice Identification

Voice ID allows the system to recognize and verify the identity of the speaker.

- When a user says a wake word, their voice is registered.
- Once registered, the user can issue commands directly without repeating the wake word.

If the speaker changes during an interaction and the new speaker is not registered (haven't said the wake word), STT processing is immediately stopped if tolerance is reached.
The tolerance for the number of times a speaker is not recognized, after the start of an interaction with registered speaker,
can be configured via : threshold_nb_chunk_unverified

**✅ Enabling Voice ID**

Use `-i vasr` with the `--voice-id` argument to enable Voice ID.

**⚙️ Voice ID Parameters**


| Parameter                       | Type    | Default | Description                                                                                                            |
| ------------------------------- | ------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `audio_chunk_duration  `        | `int`   | `3`     | Maximum duration (in seconds) of audio segments processed at a time.                                                   |
| `max_registered_users`          | `int`   | `1`     | Maximum number of registered speakers                                                                                  |
| `inactivity_timeout`            | `float` | `40.0`  | Inactivity timeout in seconds                                                                                          |
| `threshold_nb_chunk_unverified` | `int`   | `2`     | Tolerance of the number of times a speaker is not recognized after the start of an interaction with registered speaker |

**⚙️ Voice ID API**

```python
from voice_id.utils import load_audio
from voice_id.speaker import SpeakerEncoder, merge_speakers

# Initialize model
speaker_encoder = SpeakerEncoder("resnet34")

# Load audio file
audio_file_speakers = ['/path/to/your/audio_speaker_0.wav', '/path/to/your/audio_speaker_1.wav']

input_chunks1 = load_audio(audio_file_speakers[0], speaker_encoder.model_config.sample_rate)[0]
input_chunks2 = load_audio(audio_file_speakers[1], speaker_encoder.model_config.sample_rate)[0]

# Speaker 1
# verify that enough samples to send to the model
if len(input_chunks1) >= speaker_encoder.model_config.model_required_samples:
    # Send chunk to the model. Return a new speaker.
    spk1 = speaker_encoder(input_chunks1)

# Speaker 2
# verify that enough samples to send to the model
if len(input_chunks2) >= speaker_encoder.model_config.model_required_samples:
    # Send chunk to the model. Return a new speaker.
    spk2 = speaker_encoder(input_chunks2)

if spk2 ==  spk1 :
    merge_speakers(spk1, spk2)
    print("Speaker 2 is the same speaker as the speaker 1")
else : 
    print("Two different speakers")
```

⎺⎺⎺
<a name="voice-activity-detection-vad"></a>

### Voice Activity Detection (VAD)

VAD is a standalone module that detects the presence of speech in audio streams. It uses silero-vad, a lightweight neural network optimized for real-time voice activity detection.

**⚙️ VAD API**

```python
from vad.vad import VAD
from vad.utils import load_audio
from vad.user_config import UserConfig

# Initialize VAD
vad = VAD(save_audio=UserConfig.save_audio)

# Import audio file
audio_file = '/path/to/your/audio.wav'
audio_input = load_audio(audio_file, sample_rate=vad.sample_rate)
audio_input = audio_input[UserConfig.audio_channel_index]

# 1- No streaming mode: process entire audio file at once
is_speech, chunk, _ = vad(audio_input, streaming=False)
print(is_speech, chunk)

# 2- Streaming mode: process audio in chunks
vad.init_streaming_state()
for start_idx in range(0, audio_input.shape[-1], vad.required_samples):
    end_idx = min(start_idx + vad.required_samples, audio_input.shape[-1])
    chunk = audio_input[start_idx:end_idx]
    if chunk.shape[-1] < vad.required_samples:
        break
    is_speech, chunk, _ = vad(chunk)
vad.flush()
```

**⚙️ VAD configuration**

The VAD adapter is integrated into eIQ GenAI Flow via [src/eiq_genai_flow/adapters/vad.py](src/eiq_genai_flow/adapters/vad.py). VAD is automatically activated
when STT is enabled.


| Parameter                 | Type     | Default | Description                                                                                      |
|---------------------------|----------| ------- |--------------------------------------------------------------------------------------------------|
| `save_audio`              | `bool`   | `False` | Enables saving the captured audio to a WAV file.                                                 |
| `threshold`               | `float`  | `0.3`   | Voice Activity Detection threshold (0.0-1.0). Higher values require stronger speech signals.     |
| `min_silence_duration_ms` | `int`    | `200`   | Minimum silence duration (in ms) required for the VAD to detect an end of speech.                |
| `pre_vad_samples`         | `int`    | `1536`  | Number of audio samples to keep before detected speech onset (avoids cutting off initial words). |

<a name="speech-to-text-stt"></a>

### Speech-To-Text (STT)

Speech-To-Text converts spoken language into text. This project supports multiple Speech-To-Text models optimized for NXP platforms.

- moonshine-tiny
- moonshine-base
- whisper-small.en


**✅ Enabling Speech-To-Text**

Use the `--input-mode` argument with one of the following values:

- `-i vasr`: Enables Speech-To-Text after detecting the VIT Wake-Word.
- `-i kasr`: Activates Speech-To-Text via keyboard input (press "Enter" to start transcription).
- `-i keyb`: Disables Speech-To-Text, using keyboard input only.
- `-i chat_interface`: Enables a chat-like example interface for interaction, if exists. See [GUI](#graphical-user-interface-gui).
- `-i <user_gui>`: Enables the user defined interface for interaction, if it inherits GuiConfig Class. See [GUI](#graphical-user-interface-gui)

To enable Voice ID alongside STT, see [Voice Identification](#voice-identification).

To enable continuous Speech-To-Text, pass the `-c` flag. In this mode, Speech-To-Text remains active until a timeout occurs due to inactivity.

**⚙️ Speech-To-Text API**

```python
import torch
from speech_to_text.speech_to_text import SpeechToText
from speech_to_text.utils.utils import load_audio

# Initialize model
# e.g., 'whisper-small.en', 'moonshine-base', 'moonshine-tiny'
stt = SpeechToText('whisper-small.en', language='English', task='transcribe')

# Load and prepare audio
audio_file = '/path/to/your/audio.wav'
audio_input = load_audio(audio_file, sample_rate=stt.sample_rate)
audio_input = audio_input[stt.audio_channel_index]

# Split audio into chunks
if stt.model_type == 'whisper':
    input_chunks = stt.audio_processor.split(audio_input)
else:
    input_chunks = torch.split(audio_input, stt.audio_chunk_length)

# Process chunks with stt
text = ''
for chunk_idx, chunk in enumerate(input_chunks):
    ending = chunk_idx == len(input_chunks) - 1
    for text in stt(chunk, ending=ending):
        pass

print(text)
```

Speech-To-Text works in conjunction with the VAD module (see [Voice Activity Detection](#voice-activity-detection-vad))
which detects speech boundaries. The STT adapter is integrated into eIQ GenAI Flow via [src/eiq_genai_flow/adapters/stt.py](src/eiq_genai_flow/adapters/stt.py).

Some of the parameters configured in the `stt_init` override Speech-To-Text default parameters.


| Parameter              | Type    | Default | Description                                                                             |
| ---------------------- | ------- | ------- | --------------------------------------------------------------------------------------- |
| `audio_chunk_duration` | `float` | `3.`    | Maximum duration (in seconds) of audio segments processed at a time.                    |
| `max_decoded_tokens`   | `int`   | `18`    | Maximum number of tokens decoded per audio segment.                                     |
| `tokens_per_second`    | `float` | `4.`    | Expected tokens processed per second of audio.                                          |
| `stream_print`         | `bool`  | `False` | Prints decoded text during streaming.                                                   |
| `temperature`          | `float` | `0.`    | Temperature for decoder sampling (`0` is argmax).                                       |
| `prompt`               | `str `  | `''`    | Text prompt to guide the initial decoding process (only for Whisper).                   |
| `inactivity_timeout`   | `float` | `20.`   | Timeout duration (is seconds) when waiting for speech activity before interrupting STT. |

**📊 Speech-To-Text Benchmark**

Model profiling and WER evaluation are available [here](https://www.nxp.com/applications/technologies/human-machine-interface/voice-processing/speech-to-text:STT).


**🚀 NPU Acceleration (Neutron)**

This feature is experimental for the Speech-To-Text module. Enable it via `use_neutron = True` in STTAdapterConfig (adapters/stt.py).
- Only the Whisper-small.en model is supported.
- For optimal performance, run Neutron on the encoder only.
- The Whisper-small.en encoder is 20% faster when using Neutron.

NPU acceleration can be used on **i.MX95 B0** with extended CMA (> 3GB). See the [Using NPU Acceleration](#using-npu-acceleration) section for more information.

<a name="retrieval-augmented-generation-rag"></a>

### Retrieval-Augmented Generation (RAG)

RAG enhances the LLM’s responses by grounding the input in factual information from a knowledge base. This significantly improves the relevancy of the response to the prompt and reduces LLM hallucinations overall.

The demonstrator uses all-MiniLM-L6-v2 int8-quantized embedding model with 22M parameters.

**✅ Enabling RAG**

Use the `--use-rag` or `-r` argument to activate RAG.

#### RAG Example

The pre-generated RAG database is about medical healthcare for patients with diabetes, so questions related to this topic can be asked.

#### Generate your own RAG Database

To easily create a RAG database for fine-tuning and domain specific LLM-responses with GenAI Flow, please follow the `rag_database_generator` README.md.


<a name="large-language-model-llm"></a>

### Large Language Model (LLM)

The LLM is responsible for understanding input and generating relevant text-based responses. It predicts words based on the given input using advanced language modeling techniques.

#### LLM on CPU/NPU (Neutron)

The demonstrator uses Danube int8 or int4 quantized LLM with 500M parameters, derived from Llama LLM family.


**✅ Enabling LLM**

LLM is enabled by default and requires no additional parameters.
Answers given by the LLM have a maximum number of words, if this number is reached, it will print "[...]".
This limit is customizable via the `max_tokens_to_keep` setting.

**🚀 NPU Acceleration (Neutron)**

NPU acceleration can be used for LLM inference on **i.MX95 B0** with extended CMA (> 3GB). See [Using NPU Acceleration](#using-npu-acceleration).
To enable NPU acceleration for the LLM, pass the `--use-neutron` flag when running the pipeline on supported BSPs.
```bash
# Enable Neutron NPU acceleration
eiq_genai_flow --use-neutron -m danube-500M-q8
```

**⚙️ LLM Parameters**

Each LLM model has default parameters optimized for its architecture. When the LLM initializes, it logs the active parameters:

```
INFO:utils.utils - === LLM Configuration ===
INFO:utils.utils - Model: danube-500M-q8
INFO:utils.utils -   Maximum tokens: 96 (model default)
INFO:utils.utils -   Temperature: 0.25 (model default)
INFO:utils.utils -   Min-P: 0.05 (model default)
INFO:utils.utils -   End margin: 17 (model default)
INFO:utils.utils - ========================
```

**Customizing LLM Parameters:**

To override model defaults, edit the LLMConfig class in `config.py` file and set the desired values:

```python:config.py
# LLM parameters (set to None to use model-specific defaults)
temperature: float = None  # Use model default
top_k: int = None  # Use model default
top_p: float = None  # Use model default
min_p: float = None  # Use model default
repetition_penalty: float = None  # Use model default
end_margin: int = 20 # Override: use 20 instead of model default
max_tokens_to_keep: int = 128  # Override: use 128 tokens instead of model default
```

When you override parameters, the log will indicate the source:

```
INFO:utils.utils - === LLM Configuration ===
INFO:utils.utils - Model: danube-500M-q8
INFO:utils.utils - Maximum tokens: 128 (from config.py, model default was: 96)
INFO:utils.utils - Temperature: 0.25 (model default)
INFO:utils.utils - Min-P: 0.05 (model default)
INFO:utils.utils - End margin: 20 (from config.py, model default was: 17)
INFO:utils.utils - ========================
```

**Parameter Descriptions:**

- `temperature`: Controls randomness (lower = more focused, higher = more creative)
- `top_k`: Limits sampling to top K most likely tokens (used by some models)
- `top_p`: Nucleus sampling threshold - cumulative probability (used by some models)
- `min_p`: Sets a floor on token probability relative to the most likely token (used by some models)
- `repetition_penalty`: Penalty for repeating tokens - higher = less repetition (used by some models)
- `end_margin`: Token margin before forcing response termination
- `max_tokens_to_keep`: Maximum number of tokens the LLM can generate

**Note:** Different LLM models use different sampling strategies. The logging will show which parameters are active for your selected model.

**⚙️ LLM API**

Ensure that dependencies are installed (using `install.sh`). Then, you can run `python3 your_script.py`.

```python
from llm.modeling_llm import make_LLM
from llm.config.user_config import Config as user_config

# Create LLM instance
llm = make_LLM(
    name="danube-500M-q8",  # model
    user_params=user_config,  # user-defined configuration
)

while True:
    question = input("\nType your question here: ")
    for i, token in enumerate(llm(question, user_config.prompt)):
        print(token, end="")
```

Alternatively, you can refer to the `__main__.py` file located inside the installed `llm` package (in your Python `site-packages/llm/` directory) to see how the LLM system can be used as a command-line tool with various arguments and options for customization.

#### LLM on Discrete NPU (ARA2)

**ARA2** is NXP's discrete Neural Processing Unit that connects via M.2 interface, providing dedicated hardware acceleration for LLM inference on compatible i.MX platforms.

**What is ARA2?**

The [ARA2 M.2 Module](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) is a PCIe-based AI accelerator that uses the **M.2 2280 form factor** (22mm wide, 80mm long) with **Key M connector**. Key M is the standard keying for PCIe x4 NVMe devices, featuring a single notch on the right side of the connector.

**Prerequisites:**

**Hardware:**

- Compatible i.MX platform with **M.2 2280 socket, Key M** (supports PCIe):
  - i.MX95 Board with M.2 M-Key Gold Fingers (i.e. FRDM i.MX95)
  - i.MX8M Plus Board with M.2 M-Key Gold Fingers (i.e. FRDM i.MX8MPlus)
- [ARA2 M.2 Module](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) properly installed

**Note:** The M.2 socket must support **PCIe interface**. WiFi-only M.2 sockets (Key E) are **not compatible**.

**Software:**

- **BSP Version**: Linux 6.18.20_2.0.0 or later

The ARA2 stack relies on two components: `rt-sdk-ara2` and `eiq-aaf-connector`.

- **rt-sdk-ara2**: already installed as part of the BSP.
- **eiq-aaf-connector**: shipped on the platform in `/usr/share/eiq/aaf-connector/` but not installed by default. 
Install it by tuning and running the `install.sh` script provided in that directory:

```bash
cd /usr/share/eiq/aaf-connector/
# Review and tune install.sh for your setup, then run it
./install.sh
```

Verify installation:

```bash
systemctl status rt-sdk-ara2.service
dpkg -l | grep eiq-aaf-connector
```

**ARA2 LLM Models:**

1. **Get models**
   ```bash
   # Fetch models using `fetch_models` helper script:
   fetch_models --repo-id nxp/Qwen2.5-7B-Instruct-Ara240
   fetch_models --repo-id nxp/Qwen2.5-Coder-1.5B-Ara240
   ```

The models are being fetched from [NXP's Hugging Face page](https://huggingface.co/nxp) and put in `/usr/share/llm/`

2. **Configure enabled models** in `/usr/share/eiq/aaf-connector/server_config.json`:

   ```json
    {
      "available_models": [
        {
          "name": "Qwen2.5-Coder-1.5B",
          "description": "Qwen2.5-coder 1.5B instance with code generation support.",
          "type": "text",
          "tool_calling": "native",
          "enabled": true
        },
        {
          "name": "Qwen2.5-7B-Instruct",
          "description": "Qwen2.5 7B Instruct Unimodal model",
          "type": "text",
          "tool_calling": "native",
          "max_prompt_size": 2047,
          "enabled": true
        },
        {
          "name": "Qwen2.5-VL-7B-Instruct",
          "description": "Qwen2.5-VL 7B instance with vision and language capabilities.",
          "type": "qwen_vl_video",
          "tool_calling": "no",
          "max_prompt_size": 2047,
          "enabled": false
        },
      ]
    }
   ```

   **Important:**

   - Set `"enabled": true` for models you want to use
   - Only enabled models will appear in the eIQ GenAI Flow model list
   - The `name` field must match the directory name in `/usr/share/llm/`
   - ARA2 VLM models (type "qwen_vl_image") are not supported

   **Troubleshooting:**

   While using Qwen2.5-Coder-1.5B model, if you get stuck without any response for the LLM, make sure to configure the sampling parameters in `/usr/share/eiq/aaf-connector/server_config.json` as the following:
   ```json
      "llm_params": {
        ...
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        ...
      }
    ```

**✅ Using ARA2**

ARA2 models are automatically detected and suffixed with `-ara`:

```bash
# List available models (look for -ara suffix)
eiq_genai_flow --help

# Run with ARA2 model
eiq_genai_flow -m Qwen2.5-7B-Instruct-ara -i vasr -o tts -r
```

**⏱️ First Run:** The eiq-aaf-connector is launched if not yet running. Initial model loading takes **up to 10 minutes**. Subsequent runs are much faster.

**Verification:**

```bash
# Check hardware detection
lspci -nn | grep 1e58:0002

# Check service
systemctl status rt-sdk-ara2.service

# Test connector API
curl http://localhost:8000/v1/models

# Run the application with DEBUG logs to verify ARA2 connectivity
eiq_genai_flow -m Qwen2.5-7B-Instruct-ara -i vasr -o tts -r -l DEBUG
```

**Troubleshooting ARA2:**


| Issue                                         | Solution                                                                                                                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ARA2 not detected**                         | Verify module is seated properly in M.2 slot; check`lspci -nn`                                                                                                                        |
| **Service won't start**                       | `systemctl start rt-sdk-ara2.service` or reboot                                                                                                                                       |
| **No models found**                           | Check`/usr/share/llm/` directory exists and contains models, with `name` aligned in `/usr/share/eiq/aaf-connector/server_config.json` and `enabled` set to `true`;                    |
| **Connector not responding**                  | Wait up to 10 minutes on first run; check `ps aux                                                                                                                                     |
| **Connector not ready within timeout period** | check`server_config.json` validity, run the connector manually: `source /usr/share/eiq/aaf-connector/venv/bin/activate && /usr/share/eiq/aaf-connector/venv/bin/connector`, or reboot |

**View logs:**

```bash
journalctl -u rt-sdk-ara2.service -f
```

For support: [NXP Community Forum - Generative AI & LLMs](https://community.nxp.com/t5/Generative-AI-LLMs/bd-p/Generative-AI-LLMs)

⎺⎺⎺
<a name="text-to-speech-tts"></a>

### Text-To-Speech (TTS)

TTS converts the LLM-generated text responses into speech output from a speaker for audio playback.

The demonstrator uses a Vits int8-quantized model with 19.5M parameters.

**✅ Enabling TTS**

Use the `--output-mode tts` argument to enable TTS, or `--output-mode text` to disable it.

**⚙️ TTS API**

Ensure that dependencies are installed (using `install.sh`). Then, you can run `python3 your_script.py`.

```python
# script example
import os
import soundfile as sf
from tts.model import TextToSpeech
from tts.config import MultiSpeakerTTS16kHzConfig, MultiSpeakerTTS16kHzQuantConfig

# you can choose between the normal or quantized model (the latter is faster)
# config = MultiSpeakerTTS16kHzConfig(
config = MultiSpeakerTTS16kHzQuantConfig(
    speaker_id=24,  # between 1 and 904
    speed=0.52,  # the greater, the faster
    # pronunciation_json="pronunciation.json"  # OPTIONAL: JSON file path for custom pronunciation
)

# create folder for generated speech
os.makedirs("tts_gen/", exist_ok=True)

# default mode
tts = TextToSpeech(config=config)
audio_data = tts.generate("Hello world!")
# you can then save or play the generated data
sf.write("tts_gen/default.wav", audio_data, 16000)

# streaming mode
tts = TextToSpeech(config=config, mode="streaming")
audio_data = tts.generate("Hello, my name is Brian. How are you today?")
# you can then save or play the generated data
for i, chunk in enumerate(audio_data):
    sf.write(f"tts_gen/streaming_{i}.wav", chunk, 16000)

os._exit(0) # exit the program and avoid waiting for the timeout to end
```

Alternatively, you can refer to the `__main__.py` file located inside the installed `tts` package (in your Python `site-packages/tts/` directory) to see how the LLM system can be used as a command-line tool with various arguments and options for customization.

If you want to replace certain words with others that suit you better in terms of pronunciation, <br>
you can specify your own JSON in the following format:

```json
{
  "english": {
    "Stephane": "Steffen",
    "Baptiste": "Batiste",
    "Francoise": "Fronswaz",
    "CEO": "C E O",
    "VP": "Vee pee",
    "etc": "et cetera"
  }
}
```

**📊 TTS Benchmark**

Model profiling and speech quality measurement (DNS-MOS) are available [here](https://www.nxp.com/design/design-center/software/embedded-software/text-to-speech:TTS).

⎺⎺⎺

<a name="audio-manager"></a>

### Audio Manager

The Audio Manager is a core infrastructure component that provides seamless audio capture and playback with support for multiple simultaneous readers and flexible audio routing. It uses a circular master buffer that allows multiple readers (VIT, STT) to independently read the same captured audio stream with zero gaps, while managing playback through a separate queue.

![Audio Manager ](assets/audio_manager.png)

**✅ Key Features**

- **Flexible Backends**: Supports both ALSA (direct) and GStreamer backends
- **Multi-Reader Architecture**: Multiple components (VIT, STT) can independently read from the same audio stream
- **Zero-Gap Audio**: Eliminates gaps between wake-word detection and speech recognition
- **Format Conversion**: Automatic conversion between audio formats (S32LE ↔ S16LE ↔ F32LE)
- **Channel Routing**: Extract specific channels from multi-channel audio streams
- **Event-Driven**: Efficient blocking reads with condition variables (no polling)
- **Recording Support**: Optional WAV file recording for both capture and playback

**⚙️ Configuration**

The Audio Manager is configured via the `config.py` file:

```python:config.py
# Audio Capture
keep_capture_device_open: bool = False  # Keep device open continuously
save_audio_capture: bool = False        # Record captured audio to WAV

# Audio Playback
keep_playback_device_open: bool = True  # Keep device open for minimal latency
save_audio_playback: bool = False       # Record playback audio to WAV

# Recording
audio_save_path: str = "tests/recordings/"  # Directory for WAV files
```

**🔧 Backend Selection**

The Audio Manager supports two backends:

```bash
# Direct ALSA (lowest latency)
AUDIO_BACKEND=alsa eiq_genai_flow

# GStreamer (default, better device compatibility)
eiq_genai_flow
```

See example code in the `__main__.py` file located inside the installed `audio_manager` package (in your Python `site-packages/audio_manager/` directory).


⎺⎺⎺
<a name="adapters"></a>

### Adapters

Adapters bridge audio-based AI components (VIT, STT, TTS) and the GenAI Flow pipeline, providing unified thread management, state control, and audio integration.

**📋 Available Adapters**


| Adapter            | Component                | Processing Model                                           |
| ------------------ | ------------------------ | ---------------------------------------------------------- |
| **VITAdapter**     | Wake-Word Detection      | Continuous audio → Wake word event                        |
| **VoiceIDAdapter** | Voice ID                 | Continuous audio → Verification of the speaker's identity |
| **VADAdapter**     | Voice Activity Detection | Continuous audio → Speech segments                        |
| **STTAdapter**     | Speech-to-Text           | Continuous audio → Transcribed text                       |
| **TTSAdapter**     | Text-to-Speech           | Text queue → Audio chunks (streaming)                     |

---

<a name="using-npu-acceleration"></a>

## Using NPU Acceleration

NPU acceleration can be used for LLM and STT inference on i.MX 95 B0. It requires the BSP to have an extended CMA (> 3GB) for the Neutron NPU. This CMA is defined via the Linux device tree — ensure such a DTB is set as fdtfile in U-Boot; see [CMA Configuration](#cma-configuration).
- To enable Neutron for LLM: see [LLM section](#large-language-model-llm).
- To enable Neutron for STT: see [STT section](#speech-to-text-stt).
---

<a name="benchmark-mode"></a>

## Benchmark Mode

The eIQ GenAI Flow includes a benchmark mode for performance evaluation and testing any configuration of the flow.
It converts the `src/eiq_genai_flow/benchmark/data/questions.txt` file to wav files if necessary to feed the pipeline and collect performance metrics. The [questions.txt](src/eiq_genai_flow/benchmark/data/questions.txt) can be customized to match the RAG database for instance. Results are given as a json report (metrics only) and a log file (detailed traces from each pipeline stage).
This mode allows to measure key average metrics per request such as:

- **TTFA Avg**: Time To First Audio (seconds), the lower the better
- **Time Avg**: Total pipeline processing time (seconds), the lower the better
- **CPU Avg**: Average CPU utilization (%), the lower the better
- **Memory Avg**: Average memory usage (MB)
- **STT Avg Time**: Speech-To-Text processing time (seconds), the lower the better
- **RAG Avg Time**: RAG processing time (seconds), the lower the better
- **LLM Avg Time**: LLM processing time (seconds), the lower the better
- **TTS Avg Time**: TTS processing time (seconds), the lower the better
- **STT WER**: Speech-To-Text Word Error Rate (%), the lower the better
- **LLM TTFT**: Large Language Model Time To First Token (seconds), the lower the better
- **LLM TPS**: Large Language Model Tokens Per Second, the higher the better
- **TTS RTF**: Text-To-Speech Real Time Factor, the lower the better

To run benchmark mode:

```bash
eiq_genai_flow -i vasr -r -b # usual configuration + '-b'
```

Various configurations have been benchmarked, results are available in [eIQ GenAI Flow Page](https://www.nxp.com/applications/technologies/human-machine-interface/voice-processing/simplified-and-optimized-generative-ai-at-the-edge-with-eiq-genai-flow:GEN-AI-FLOW).

---

<a name="audio-setup"></a>

## Audio setup

The demonstrator can use on-board codecs when present (i.e. on EVKs [WM8962 codec](https://community.nxp.com/pwmxy87654/attachments/pwmxy87654/imx-processors/58279/1/WM8962_v4.2.pdf) which manages both input and output through a single 3.5mm jack connector CTIA), or USB devices on FRDM boards for capture/playback, to be used with some precautions, see below.

### WM896* codec

This codec is the one chosen by default when the `--capture-device` and `--playback-device` parameters are not given.

To use the audio functionalities, the following setups are possible:

- **🎧 Headset Mode:** Use a headset with an integrated microphone and a 4-pole CTIA connector.
- **🔊 Open Audio Setup:** Use a **3.5mm jack audio splitter** (4-pole CTIA) along with:
  - 🎤 A standalone **microphone** (3-pole)
  - 🔉 A **loudspeaker**

Setup example:

![Complete demo setup](assets/demo_setup.png)

This ensures proper handling of both input and output audio during the demonstrator's operation.

Some settings defined in the shared_utils/audio_config.sh such as the capture level and playback level are automatically applied on startup, and can be customized in this shell script.

Note that if this codec is not present on the platform, a fall-back to another device will be done, as displayed while passing the `-h` to the project.

### USB devices/codecs

Running the demonstrator with `-h` will display the audio interfaces found on the system, and selectable via the `--capture-device` and `--playback-device` parameters.

The displayed interfaces use ALSA's "plughw" or "sysdefault" format, which automatically converts audio formats to match your hardware requirements.

**Important Notes:**

- All audio devices found by the BSP will be listed, however not all devices may be compatible with the required audio format
- Audio devices must support FLOAT_LE format, 2 channels, 16kHz (the ALSA plug plugin handles format conversion automatically)
- Try both "plughw" and "sysdefault" formats - one may work better with your hardware. Test with the audio commands in the the [Troubleshooting](#troubleshooting) section or enable recording in `config.py` (`save_audio_capture`, `save_audio_playback`) to validate your setup.
- For any audio issues, see the [Troubleshooting](#troubleshooting) section

**Customization:**
The audio settings (volume, etc..) can be customized by editing the [set_audio_device_config.py](audio_manager/src/audio_manager/set_audio_device_config.py) script from the `audio_manager`, which can automatically applies a custom configuration when the demonstrator starts.

⎺⎺⎺
<a name="graphical-user-interface-gui"></a>

### Graphical User Interface (GUI)

The eIQ GenAI Flow automatically searches for python apps which inherits the GuiConfig class, and placed in gui/modules. The GUIs discovered are listed in the input configurations (`-i`).
The eIQ GenAI Flow includes a chat interface example for interactive conversations with visual feedback.

The GUI runs as a separate process and communicates through message queues for responsive interaction during AI processing.

**🚀 Quick Launch**

The chat interface is installed either by running `./install.sh --gui` or by directly installing the `nxp_eiq_chat_interface` wheel:

```bash
./install.sh --gui
```

OR

```bash
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
pip install wheels/py$PY_VERSION/nxp_eiq_chat_interface-*
```
The chat_interface is started separately, then eIQ GenAI Flow must be started with the `chat_interface` input mode, which is a `vasr` mode + gui.
The start GUI mode:

```bash
chat_interface&  # start the gui
eiq_genai_flow -i chat_interface --voice-id -r  # vasr mode + gui, and usual customizable arguments

```

**💬 Features**

- Real-time chat bubbles for questions and responses
- Visual status indicators for connection and listening states
- Streaming AI responses as they generate

⎺⎺⎺

<a name="ros-node"></a>

## ROS 2 Node

`ros_node.py` provides a ROS 2 wrapper around the eIQ GenAI Flow pipeline. It exposes the full pipeline as a standard ROS 2 node, publishing events (wake-word, VAD, STT transcription, LLM tokens, TTS state, Voice ID) to dedicated topics and accepting text queries or trigger commands via a topic and a service.

This makes it straightforward to integrate eIQ GenAI Flow into any ROS 2-based robot or application without modifying the core pipeline.

For installation instructions, topic/service reference, and usage examples, see the [ROS README](ROS_README.md).

---

<a name="other-customizations"></a>

### Other customizations

The demonstrator package includes a `config.py` file that allows for extensive customization of the eIQ GenAI Flow behavior and parameters.

**Key Configuration Parameters:**

**Interfaces & Paths:**

- `tests_data_path`: Directory holding test data (default: "tests/data")
- `benchmark_questions_file`: File with questions used in benchmark mode (default: "tests/data/questions.txt")
- `update_global_benchmark_json`: Update the global benchmark JSON report on each run (default: `False`)

**System Messages & Prompts:**

- Various configurable system messages and prompts exist (e.g. the default LLM system prompt, TTS greeting, console prompt, STT/Voice ID startup banners, and listening indicator). See the configuration source for the full list and defaults.

**Performance & CPU Settings:**

- `set_cpu_governor`: Enable/disable CPU governor configuration (default: `True`)
- `cpu_governor`: CPU governor mode (default: "performance")
- `restore_cpu_governor_on_exit`: Restore original CPU governor on exit (default: `True`)

**RAG parameters:**

- `rag_db_name`: Name of the RAG database file (default: "medical_db.pkl")
- `rag_db_path`: Full path to the RAG database file (default: retriever's default database)
- `similarity_threshold`: Minimum similarity score for RAG retrieval (default: 0.65)

**Audio Feedback:**

- `play_tts_start_sound`: Play notification sound before TTS speaks (default: `True`)
- `play_wake_word_detect_sound`: Play sound when wake-word is detected (default: `True`)
- `play_intent_detect_sound`: Play sound when an intent is detected by RAG (default: `True`)

**Audio Recording:**

- `save_audio_capture`: Record captured audio to a WAV file (default: `False`)
- `save_audio_playback`: Record playback audio to a WAV file (default: `False`)
- `save_audio_vit`: Enable audio saving for VIT wake-word detection (default: `False`)
- `audio_save_path`: Directory where recorded WAV files are stored (default: "tests/recordings/")

**Audio Devices:**

- `keep_playback_device_open`: Keep playback device always open for minimal latency (default: `True`)
- `keep_capture_device_open`: Keep capture device always open; `False` opens it only when VIT/ASR is active (default: `True`)

**Audio Fillers (generated by the TTS):**

- `play_audio_filler`: Play a filler sentence while processing (default: `False`)
- `audio_filler_path`: Directory holding the filler audio assets (default: "assets/fillers")
- `audio_filler_sentences`: List of filler sentences to play during processing (default: `["Okay.", "Got it.", "Alright.", "Let's see.", "One moment.", "Just a second."]`)

**Benchmark Settings:**

- `silent_benchmark`: Disable TTS playback and notifications during benchmark mode (default: `True`)

**TTS Settings:**

- `tts_mode`: TTS mode, "streaming" or "default" (default: "streaming")
- `tts_speed`: TTS speaking speed; higher is faster (default: 0.55)
- `tts_speaker_id`: TTS speaker voice ID (default: 24)

**DNPU Acceleration Service:**

- `discrete_npu_service`: systemd service name for the discrete NPU (ARA2) acceleration (default: "rt-sdk-ara2.service")

**VIT Wake-word Settings:**

- `wake_model_path`: Custom VIT wake-word model path; empty string uses the default from the vit package (default: "")
- `vit_channel_indices`: Audio channel indices used by VIT (default: `[0]`)

**STT (ASR) Settings:**

- `stt_channel_indices`: Audio channel indices used by STT (default: `[0]`)
- `stt_inactivity_timeout`: Timeout in seconds for STT inactivity (default: 15)

**LLM Parameters (`LLMConfig`):**

Set to `None` to use model-specific defaults. Note: these parameters are NOT used for ARA LLMs (models ending with `-ara`), which are controlled by `/usr/share/eiq/aaf-connector/server_config.json`.

- `temperature`: Controls randomness of generation (default: `None`, model default)
- `top_k`: Limits sampling to top K tokens (default: `None`, model default)
- `top_p`: Nucleus sampling threshold (default: `None`, model default)
- `min_p`: Floor on token probability relative to the most likely token (default: `None`, model default)
- `repetition_penalty`: Penalty for repeating tokens (default: `None`, model default)
- `end_margin`: Token margin before forcing response termination (default: 20)
- `max_tokens_to_keep`: Maximum number of tokens the LLM can generate (default: 128)


**GUI:**

- `available_gui_list`: List of selectable GUI modules (default: `["chat_interface"]`)

**Example customization:**


```python:config.py
# Use a more specific system prompt
default_system_prompt: str = "You are a medical assistant specializing in diabetes care."

# Increase similarity threshold for more precise RAG matching
similarity_threshold: float = 0.75

# Disable audio notifications
play_tts_start_sound: bool = False
play_wake_word_detect_sound: bool = False
play_intent_detect_sound: bool = False
```

Edit the `config.py` file directly to customize these parameters according to your application requirements.

---

<a name="troubleshooting"></a>

## Troubleshooting

### Audio Issues

eIQ GenAI Flow opens the audio devices in FLOAT_LE format, 2 channels, 16kHz. It uses the Alsa plug plugin to make relevant format conversions.
If you're experiencing audio problems with the eIQ GenAI Flow, use the following commands to diagnose and test your audio setup:

```bash
# List available audio devices
arecord -l  # Input devices
aplay -l    # Output devices

# Test audio recording and playback
arecord -D plughw:CARD=CAPTURE_DEVICE -d 10 -r 16000 -c 2 -f FLOAT_LE > out.pcm
aplay -D plughw:CARD=PLAYBACK_DEVICE -d 10 -r 16000 -c 2 -f FLOAT_LE out.pcm
```

*Note: Replace "CAPTURE_DEVICE" and "PLAYBACK_DEVICE" with the actual device names returned by `aplay -l` and `arecord -l` commands.*

**Common Audio Issues:**

- **No audio devices found**: Ensure your audio hardware is properly connected and recognized by the system
- **Permission denied**: You may need to add your user to the `audio` group: `sudo usermod -a -G audio $USER`
- **Device busy**: Another application might be using the audio device. Close other audio applications or restart the system
- **Poor audio quality**: Check cable connections and ensure you're using the correct audio format settings
- **No Audio**: Your device, plug plugin or BSP may not be compatible with eIQ GenAI Flow audio configuration

For USB audio devices, refer to the [Audio setup](#audio-setup) section for additional configuration guidance.

### NPU Issues

Neutron NPU acceleration for LLM is only available on i.MX95 B0, on compatible BSP versions with compatible ONNXruntime library installed and extended CMA. Diagnostics and troubleshooting steps:

<a name="cma-configuration"></a>

#### CMA configuration

```bash
# Check CMA memory allocation:
cat /proc/meminfo | grep -i CMA
```

It must be > 3GB: 2GB is required for the NPU and 1GB for the system.

If not, the Neutron device tree blob (DTB) must be set manually in U-Boot.
To set the correct DTB neutron in u-boot:

* Stop the boot at u-boot stage (press a key)
* List and identify the available Neutron DTB files. If flashed on an SD card, use:
  ```
  u-boot=> fatls mmc 1:1
  ```
* Set a Neutron-enabled DTB, for example:

```bash
u-boot=> setenv fdtfile imx95-19x19-evk-neutron.dtb
u-boot=> saveenv
u-boot=> boot
```

If no neutron dtb are available, build your own from the Linux kernel: Append imx95-15x15-evk-neutron.dtbo or imx95-19x19-evk-neutron.dtbo to your current dtb in `arch/arm64/boot/dts/freescale/Makefile`, and build it.
For instance on i.MX95 FRDM board:

```
imx95-15x15-frdm-neutron-dtbs := imx95-15x15-frdm.dtb imx95-15x15-evk-neutron.dtbo
dtb-$(CONFIG_ARCH_MXC) += imx95-15x15-frdm-neutron.dtb
```

#### ONNX runtime Issues

The ONNX runtime version to use is the one delivered in the BSP. Any `pip install onnxruntime=="a different version"` would force the project to use a version non-compatible with Neutron.

---

<a name="support"></a>

## Support

For more general technical questions, use the [NXP Community Forum Generative AI & LLMs](https://community.nxp.com/t5/Generative-AI-LLMs/bd-p/Generative-AI-LLMs).

---

<a name="release-notes"></a>

## Release Notes

| Version | Description / Update                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Date                          |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 1.0     | Initial release on Application Code Hub for i.MX95. This is solely for evaluation and development in combination with an NXP Product.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | March 31<sup>th</sup> 2025    |
| 1.1     | Add i.MX8MP Support. This is solely for evaluation and development in combination with an NXP Product.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | June 20<sup>th</sup> 2025     |
| 2.0     | **Component release:** Pipeline and RAG as Python files, other components (STT, LLM, TTS) as binary libs. **Platform support:** i.MX9x and i.MX8Mx. **Key features:** Neutron acceleration (i.MX95 B0), customizable VIT Wake-Word, flexible audio devices, 900+ TTS voices, WhatsApp-style GUI. **Models:** moonshine-tiny/base/whisper-small.en (STT), Danube-500M q4/q8 (LLM), all-MiniLM-L6-v2 (embedding), vits-english (TTS). **Limitation:** 60-minute timeout. This is solely for evaluation and development in combination with an NXP Product.                                                                                                                                                                                      | November 21<sup>th</sup> 2025 |
| 3.0     | **Latency improvement:** New TTS streaming mode providing up to 30% improvement for TTFA. **Enhancements:** New VIT wake-word engine v4.13 with wake-word detection performance improvement and false positive robustness. **Seamless interaction:** New Audio Manager component removing gap between wake-word and STT, supports direct ALSA or GStreamer audio backends. **Limitation:** 60-minute timeout. This is solely for evaluation and development in combination with an NXP Product.                                                                                                                                                                                                                                                   | March 31<sup>st</sup> 2026    |
| 3.1     | **Latency improvement:** Improved TTS streaming mode providing up to 0.5s improvement for TTFA. **New module:** New Voice ID module which allows the system to recognize and verify the identity of the speaker. **ARA240 DNPU LLM support:** On compatible platforms, the ARA240 LLMs can be used. **Event based pipeline:** Modules are synchronized via events. **Modules as Wheels:** All the modules are delivered as wheels. **ROS2 node:** Add ROS 2 wrapper for eIQ GenAI Flow to provide a node-based interface to pipeline, allowing seamless integration with ROS 2 ecosystems. **Limitation:** 60-minute timeout. This is solely for evaluation and development in combination with an NXP Product. | July 31<sup>st</sup> 2026     |


---

<br>
<p align="center">
  <a href="https://www.nxp.com">
    <img src="https://mcuxpresso.nxp.com/static/icon/nxp-logo-color.svg" width="100"/>
  </a>
</p>
