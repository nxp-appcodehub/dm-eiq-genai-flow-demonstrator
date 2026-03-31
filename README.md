# NXP® eIQ® GenAI Flow Demonstrator Package

[![License badge](https://img.shields.io/badge/License-Proprietary-red)](./LICENSE.txt)
[![Board badge](https://img.shields.io/badge/Board-i.MX95-blue)](https://www.nxp.com/products/i.MX95)
[![Board badge](https://img.shields.io/badge/Board-i.MX943-blue)](https://www.nxp.com/products/i.MX94)
[![Board badge](https://img.shields.io/badge/Board-i.MX93-blue)](https://www.nxp.com/products/i.MX93)
[![Board badge](https://img.shields.io/badge/Board-i.MX91-blue)](https://www.nxp.com/products/i.MX91)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MPLUS-blue)](https://www.nxp.com/products/I.MX8MPLUS)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MMINI-blue)](https://www.nxp.com/products/I.MX8MMINI)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MNANO-blue)](https://www.nxp.com/products/I.MX8MNANO)

[![Language badge](https://img.shields.io/badge/Language-Python-yellow)]()
[![Category badge](https://img.shields.io/badge/Category-AI/ML-green)]()

---

## Overview

The **NXP eIQ® GenAI Flow Demonstrator** package showcases advanced AI capabilities running at the edge on **NXP i.MX9 and i.MX8M** application processors. This demonstrator includes two powerful AI applications:

1. **[eIQ GenAI Flow](eiq_genai_flow/README.md)** - Conversational AI Pipeline
2. **[Vision Language Model (VLM)](vlm/README.md)** - Visual Question Answering

Both applications are optimized for edge deployment and demonstrate real-world AI use cases on resource-constrained devices.

---

## 📦 Package Contents

### 1. eIQ GenAI Flow - Conversational AI

A complete conversational AI pipeline integrating:
- **Wake-Word Detection** (VIT)
- **Speech-to-Text** (Whisper, Moonshine)
- **Retrieval-Augmented Generation** (RAG)
- **Large Language Model** (Danube 500M)
- **Text-to-Speech** (VITS)

**📖 Detailed Documentation:** [eiq_genai_flow/README.md](eiq_genai_flow/README.md)

**Key Features:**
- Multi-turn conversations with wake-word activation
- Knowledge-base enhanced responses via RAG
- Multiple input modes (voice, keyboard, GUI)
- NPU acceleration support (i.MX95)
- 900+ TTS voice options

---

### 2. Vision Language Model (VLM)

Visual question answering system for image understanding:
- Multimodal AI combining vision and language
- Image captioning and visual reasoning
- Edge-optimized inference

**📖 Detailed Documentation:** [vlm/README.md](vlm/README.md)

**Key Features:**
- Describe and analyze images
- Answer questions about visual content
- Optimized for edge deployment

---

## 🚀 Quick Start

### Prerequisites

- **Hardware:** NXP i.MX95, i.MX943, i.MX93, i.MX91, i.MX8MP, i.MX8MM, or i.MX8MN board
- **OS:** NXP Linux BSP (L6.12.49-2.2.0 or later recommended for i.MX95)
- **Python:** Version 3.13
- **Storage:** At least 16GB free space

### Installation

1. **Transfer the package to your i.MX device:**

   ```bash
   scp -r dm-eiq-genai-flow-demonstrator root@<imx-device-ip>:/root/
   ```

2. **Install eIQ GenAI Flow:**

   ```bash
   cd eiq_genai_flow
   ./install.sh
   ```

   See [eiq_genai_flow/README.md](eiq_genai_flow/README.md#installation) for detailed installation options.

3. **Install VLM:**

   ```bash
   cd ../vlm
   ./install.sh
   ```

### Running the Demonstrators

**Conversational AI:**
```bash
cd eiq_genai_flow
python3 eiq_genai_flow.py -i vasr -o tts -m danube-500M-q8
```

**Vision Language Model:**
```bash
cd vlm
./launch.sh
```

---

## 📋 Platform Support

| Platform | eIQ GenAI Flow | VLM | NPU Acceleration |
|----------|----------------|-----|------------------|
| i.MX95   | ✅ Full        | ✅  | ✅ (LLM)        |
| i.MX943  | ✅ Full        | ✅  | ❌              |
| i.MX8MP  | ✅ Full        | ✅  | ❌              |
| i.MX93   | ✅ Partial*    | ✅  | ❌              |
| i.MX91   | ✅ Minimal**   | ✅  | ❌              |
| i.MX8MM/MN | ✅ Minimal** | ✅  | ❌              |

\* Lighter models recommended (danube-500M-q4, moonshine-tiny)  
\** RAG-only mode (no LLM generation)

See [platform recommendations](eiq_genai_flow/README.md#flow-configuration-recommendations) for detailed configuration guidance.

---

## 📚 Documentation

- **[eIQ GenAI Flow Documentation](eiq_genai_flow/README.md)** - Complete conversational AI setup and usage
- **[VLM Documentation](vlm/README.md)** - Vision language model guide
- **[License Information](LICENSE.txt)** - Terms and conditions
- **[SBOM](SBOM-eIQ-GenAI-Flow_v3.0.spdx.json)** - Software Bill of Materials

---

## 🔧 Configuration

Each component has its own configuration:

- **eIQ GenAI Flow:** Edit `eiq_genai_flow/config.py`
- **VLM:** See `vlm/README.md` for configuration options

---

## 📊 Benchmarks

Performance benchmarks for various configurations are available at:
- [eIQ GenAI Flow Benchmarks](https://www.nxp.com/applications/technologies/human-machine-interface/voice-processing/simplified-and-optimized-generative-ai-at-the-edge-with-eiq-genai-flow:GEN-AI-FLOW)

---

## ⚠️ Demonstrator Limitations

This demonstrator package has the following limitations:

- **Session timeout**: Applications automatically shut down after 1 hour of operation
- **Language support**: English language only
- **Component delivery**: Core AI components provided as optimized binary libraries
- **Model selection**: Includes a curated subset of models optimized for target platforms
- **Model format**: Models delivered in encrypted format

These limitations are designed to provide an optimal evaluation experience while showcasing AI capabilities on NXP platforms.

---

## 🆘 Support

- **Community Forum:** [NXP Community - Generative AI & LLMs](https://community.nxp.com/t5/Generative-AI-LLMs/bd-p/Generative-AI-LLMs)
- **Technical Questions:** Post on the NXP Community Forum
- **Documentation:** See component-specific README files

---

## 📄 License

This software is proprietary to NXP and may only be used strictly in accordance with the applicable license terms.

See [LICENSE.txt](LICENSE.txt) for complete terms and conditions.

### Third-Party Licenses

- See individual component licenses in the `licenses/` directory

---

## 📝 Release Notes

| Version | Release Date | Highlights |
|---------|--------------|------------|
| 3.0 | March 31, 2026 | TTS streaming, new Audio Manager, module customization, VLM demonstrator v1.0 |
| 2.0 | November 21, 2025 | Neutron acceleration, customizable wake-word, 900+ TTS voices |
| 1.1 | June 20, 2025 | i.MX8MP support |
| 1.0 | March 31, 2025 | Initial release (i.MX95) |

See component-specific README files for detailed release notes.

---

## 🎯 Use Cases

### Conversational AI Applications
- Smart home voice assistants
- Automotive in-cabin assistants
- Healthcare patient interfaces
- Industrial HMI systems

### Vision Applications
- Visual inspection and QA
- Assistive technologies
- Educational tools
- Content analysis

---

## 🏗️ Repository Structure

```
dm-eiq-genai-flow-demonstrator/
├── eiq_genai_flow/          # Conversational AI pipeline
│   ├── README.md           # Detailed documentation
│   ├── config.py           # Configuration file
│   └── eiq_genai_flow.py   # Main application
├── vlm/                    # Vision Language Model
│   └── README.md           # VLM documentation
├── licenses/               # Third-party licenses
├── LICENSE.txt            # Main license
├── SBOM-*.spdx.json       # Software Bill of Materials
└── README.md              # This file
```

---

## 🔗 Related Resources

- [NXP i.MX Application Processors](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors:IMX_HOME)
- [NXP eIQ ML Software](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ)
- [i.MX Linux BSP](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX)

---

<br>
<p align="center">
  <a href="https://www.nxp.com">
    <img src="https://mcuxpresso.nxp.com/static/icon/nxp-logo-color.svg" width="100"/>
  </a>
</p>

<p align="center">
  <strong>NXP Semiconductors - Securing the Connected World</strong>
</p>
\
