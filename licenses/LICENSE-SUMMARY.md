# License Summary for eIQ GenAI Flow v2.0

This document provides a summary of all licenses used in the eIQ GenAI Flow project.

## Main Project License

**The entire eIQ GenAI Flow project is licensed under the NXP Online Code Hosting Software License v1.4 (May 2025).**

See the main [LICENSE.txt](../LICENSE.txt) file at the root of the repository for the complete license terms.

## Component License Distribution

| License | Count | Packages/Components |
|---------|-------|---------------------|
| Apache-2.0 | 10 | accelerate, Cython, onnx, optimum, sentence-transformers, sentencepiece, transformers, whisper-small.en_onnx_NXP, h2o-danube3-500m-chat_onnx_NXP, all-MiniLM-L6-v2_onnx_NXP |
| BSD-3-Clause | 7 | NumPy, posix-ipc, psutil, pytorch, scikit-learn, soundfile, colorama |
| MIT | 6 | onnxruntime, python3-rich, typer, silero-vad, moonshine-tiny.onnx_NXP, moonshine-base.onnx_NXP |
| NXP Software License (Online Code Hosting v1.4) | 2 | eIQ-GenAI-Flow_Source, VIT_App (wrapper) |
| NXP Software License (v58 November 2024) | 1 | VIT library (embedded in VIT_App) |
| BSD-2-Clause | 1 | Pygments |
| PSF-2.0 | 1 | PyAlsaAudio |
| Mixed (BSD-2-Clause + Public Domain) | 1 | pycryptodome |

## NXP Proprietary Licenses

### Project-Level Licensing
The overall project is governed by the **NXP Online Code Hosting Software License v1.4 (May 2025)**, which covers the distribution and use of the entire eIQ GenAI Flow package.

### VIT_App Component Licensing
The VIT_App component contains multiple licensing layers:

- **VIT_App wrapper**: Licensed under NXP Online Code Hosting Software License v1.4 (May 2025)
- **VIT library (embedded)**: Licensed under NXP Software License v58 (November 2024) with clause 2.3 distribution rights

This dual licensing structure reflects that the VIT library is a separate NXP component integrated into the VIT_App wrapper application.

## License Files

### Main License
- `../LICENSE.txt` - **Main project license** (NXP Online Code Hosting License v1.4)

### Component Licenses
- `Apache-2.0.txt` - Apache License Version 2.0
- `BSD-3-Clause.txt` - BSD 3-Clause "New" or "Revised" License
- `BSD-2-Clause.txt` - BSD 2-Clause "Simplified" License
- `MIT.txt` - MIT License
- `PSF-2.0.txt` - Python Software Foundation License Version 2
- `Public-Domain.txt` - Public Domain dedication
- `NXP-Software-License.txt` - NXP Online Code Hosting License v1.4 (May 2025)
- `NXP-Software-License-v58.txt` - NXP Software License v58 (November 2024)

## Notes

- **The entire project is licensed under NXP Online Code Hosting License v1.4**
- The majority of dependencies (23 out of 25) use open-source licenses
- Core eIQ GenAI Flow components use NXP's proprietary licenses
- The VIT_App has dual licensing due to embedded VIT library component
- All open-source licenses are permissive (no copyleft requirements)

## Compliance Considerations

When distributing or using this software:
1. **Primary obligation**: Comply with the main NXP Online Code Hosting License v1.4
2. Comply with both NXP license versions for VIT-related components
3. Ensure proper attribution for all open-source dependencies
4. Review distribution rights in NXP License v58 clause 2.3 for VIT library usage

For detailed license terms, refer to the individual license files in this directory and the main LICENSE.txt file.

