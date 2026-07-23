# Vision Language Model

[![License badge](https://img.shields.io/badge/License-Proprietary-red)](./eiq_genai_flow/LICENSE.txt)
[![Board badge](https://img.shields.io/badge/Board-i.MX95-blue)](https://www.nxp.com/products/i.MX95)
[![Board badge](https://img.shields.io/badge/Board-I.MX943-blue)](https://www.nxp.com/products/i.MX94)
[![Board badge](https://img.shields.io/badge/Board-i.MX93-blue)](https://www.nxp.com/products/i.MX93)
[![Board badge](https://img.shields.io/badge/Board-i.MX91-blue)](https://www.nxp.com/products/i.MX91)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MPLUS-blue)](https://www.nxp.com/products/I.MX8MPLUS)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MMINI-blue)](https://www.nxp.com/products/I.MX8MMINI)
[![Board badge](https://img.shields.io/badge/Board-i.MX8MNANO-blue)](https://www.nxp.com/products/I.MX8MNANO)

[![Language badge](https://img.shields.io/badge/Language-Python-yellow)]()
[![Category badge](https://img.shields.io/badge/Category-AI/ML-green)]()

**VLM** submodule combines the abilities of vision and language models to handle both image and text on the **NXP [i.MX9](https://www.nxp.com/products/iMX9-PROCESSORS) applications processors.

---

![SmolVLM256M-fp32-delivery_q3.gif](assets/SmolVLM256M-fp32-delivery_q3.gif)

---

## Installation
### Set up dependencies
```bash
cd vlm
./install.sh
```

## Run VLM with Chat Interface GUI
Command to run the VLM and GUI.
```bash
# Run VLM
./launch.sh
```
It runs the chat_interface and the main vlm process. The first time you run the app it will take longer due to download of models.

- **`-m`, `--model`**  
  Specifies the VLM to use. Available models are:
  - `smolvlm-256M`
  - `smolvlm-500M`


- **`-im`, `--input_image`**  
  Path to the image to caption.

Default image delivery and industry in `test/data`

- **`-p`, `--precision`**  
Precision of model.
  - `fp32`
  - `q8`

User can choose which part of the model is fp32 vs q8 by changing config.py
  
- `-g`
Use GUI. Default True.


```bash
#Example
 ./launch.sh -m smolvlm-500M -im /path/to/your_image.png -p q8 -g
```
```bash
#Helper
 ./launch.sh --help
```


### Run without GUI
It is posible to run the code without the GUI interface.

```bash
python3 -m vlm
```

---
## Performance on i.MX95 (CPU)

| i.MX95            | Precision  | Vision Encoder | Decoder (TTFT) | Decoder       | 
|-------------------|------------|----------------|----------------|---------------|
| SmolVLM2-256M | FP32 | 6.66s          | 0.84s          | 0.13s - 0.16s |
|  | INT8| 3.31s          | 0.48s          | 0.08s - 0.09s | 
| SmolVLM2-500M | FP32| 6.76s          | 1.98s          | 0.21s - 0.25s |
|  | INT8| 3.34s          | 0.81s          | 0.12s - 0.19s |  

> SmolVLM2-256 and SmolVLM2-500M share the same vision encoder so performance are the same.


## Performance on i.MX95 (CPU+NPU)
| i.MX95        | Precision | Vision Encoder | Decoder (TTFT) | Decoder        | 
|---------------|-----------|----------------|----------------|----------------|
| SmolVLM2-500M | INT8      | 2.198s         | 0.58s          | 0.07s - 0.125s |


