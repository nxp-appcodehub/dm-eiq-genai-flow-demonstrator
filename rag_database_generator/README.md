![](assets/RAG.png)

# RAG Database Generation

[![License badge](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)
[![Board badge](https://img.shields.io/badge/Board-i.MX_95-blue)](https://www.nxp.com/products/iMX95)
[![Language badge](https://img.shields.io/badge/Language-Python-yellow)]()
[![Category badge](https://img.shields.io/badge/Category-AI/ML-green)]()

---
## Overview
<div style="text-align: justify;">

**Retrieval-Augmented Generation (RAG)** enhances the knowledge base of a Large Language Model (LLM) for a target domain, device, or context. RAG helps control the LLM’s behavior, mitigate hallucinations and customize responses based on examples. RAG involves injecting additional contextual information into the input prompt of the LLM, retrieved from a compact knowledge database stored on the edge device. 

1. **Database Generation**: A knowledge database is created offline, consisting of pairs of chunks (pieces of text) and their corresponding embeddings (vectorial representations from an embedding model).
2. **Retrieval**: Each query is encoded into an embedding using the same model. The query embedding is then compared to the database embeddings and the most similar chunks are retrieved.
3. **Generation**: A LLM receives a prompt enriched with the retrieved chunks and generates an answer.

> ⚠️ **Platform Note:** This tool is designed to run on a PC (x86) for offline database generation. The generated RAG database (.pkl file) can then be used on NXP i.MX9 and i.MX8M application processors.
</div>

---
## Features

### Database generator
* 📚 Advanced PDF document parsing leveraging [Docling](https://github.com/DS4SD/docling "Go to Docling repository").
* ✂️ Solutions to break down large textual data using chunking algorithms. Including [HiRAG](https://openreview.net/forum?id=cWWb9cgSVi "HiRAG paper") that offers an advanced chunk reformatting.
* 🧩 Embedding model adapted for NXP platforms to generate RAG databases.

### Notebooks
* 🔍 An example of the database generation process.
* 🤖 An example of using RAG with [Hugging Face](https://huggingface.co/) LLMs (TinyLlama, Llama, Gemma, Danube).

---

## Installation


<a name="set-up-the-environement"></a>
### Set up the environement:
```bash
# Install Runtime Dependencies:
./run.sh --install
```
> Works on Linux environment. This package has been tested thoroughly with [Python 3.13 and 3.14](https://www.python.org/downloads/).

#### Setting Up a Hugging Face Environment: 

1. [Create a Hugging Face account](https://huggingface.co/join).
2. [Generate a personal read access token](https://huggingface.co/settings/tokens).
3. [Log in to Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/en/guides/cli):
4. You might have to ask for access to some models on their Hugging Face page (e.g. [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct))
```bash
# Replace $HF_TOKEN with your personal access token
hf auth login --token $HF_TOKEN  
```

---

## Configurable parameters

Various parameters in the [config.py](src/rag_database_generator/config.py) file are available to customize the behavior of the project.  

| Parameter              | Type   | Default               | Description                                                                                                                     |
|------------------------|--------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------| 
| `output_format`        | `str`  | `.md`                 | Defines the output format of the PDF file after parsing.                                                                        |
| `chunking_method`      | `str`  | `HiRAG`               | Defines the strategy used for chunking the file. See all options in: [chunk.py](src/rag_database_generator/chunk.py)            |
| `chunk_size`           | `int`  | `128`                 | Defines the length (in characters) of each text chunk. Larger chunks may improve response quality but increase processing time. |
| `chunk_overlap`        | `int`  | `64`                  | Defines the overlap between consecutive chunks, typically set to half the `chunk_size` to preserve context across chunks.       |
| `embedding_model`      | `str`  | `all-MiniLM-L6-v2`    | Defines the embedding model used for the embedding computing. Run ```./run.sh --see-models``` to see all options in:       |
| `use_onnx_model`       | `bool` | `True`                | Use the ONNX version of the embedding model.                                                                                    |
| `use_quant_model`      | `bool` | `False`               | Use the quantized version ONNX version of the embedding model.                                                                  |
| `database_name`        | `str`  | `"my_first_database"` | Database file name after generation.                                                                                            |
| `database_description` | `str`  | `"..."`               | Database description that will be embedded in the RAG database during its generation.                                           |

---

## Generate Your Custom Database

These are the steps for creating a RAG database.

💡 To understand the workflow and its features, example files in the [data folder](data) are included. These files illustrate the entire process, from [parsing](data/parsed_files) and [chunking](data/chunked_files) the [Medical.pdf](data/input_files/Medical.pdf) to incorporating [handcrafted chunks](data/chunked_files/hand_made_chunks.json). The resulting database can then be used in the `eIQ GenAI Flow Demonstrator`.

### 0. (Optional) Parse PDF Files

To parse PDFs into Markdown files, we use [Docling](https://github.com/docling-project/docling/tree/main).

> **Note:** [Set up a Hugging Face environment](#optional-setting-up-a-hugging-face-environment-) may be needed before running Docling.
> 
> If any issues occur, please refer to the [Docling documentation](https://docling-project.github.io/docling/).

📂 **Where to Place PDF files**

Move the PDF files into the `input_files` folder:
```
.
├── assets
├── data
│   ├── chunked_files
│   ├── databases
│   ├── input_files
│   │   └── 📄 <<< PUT PDF FILES HERE >>>
│   └── parsed_files
├── wheels
├── LICENSE
├── notebooks
├── pyproject.toml
├── README.md
├── run.sh
└── src
    └── rag_database_generator
        ├── chunk.py
        ├── config.py
        ├── utils.py
        ├── embed.py
        ├── hirag
        ├── __init__.py
        ├── __main__.py
        └── parse.py
```

To process PDFs and extract text, run:
```bash
parse-files path/file1.pdf path/file2.pdf ...
```
> ⚠️ **Tables are not yet supported**
>
> Use the `--help` flag to see available options for `parse-files`.

### 1. Generate Chunks

To generate text chunks from Markdown files: 
```bash
chunk-files path/file1.md path/file2.json ...
```
> Use the ``--help`` flag to see available options for `chunk-files`.

> By default, the system uses the [HiRAG](https://openreview.net/forum?id=cWWb9cgSVi "HiRAG paper") chunking method, which requires running an LLM. 
> 
> To use **HiRAG**, ensure that:
>
> ✅ Access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) was requested.<br>
> ✅ [A Hugging Face environment is set up](#optional-setting-up-a-hugging-face-environment-).<br>
> ✅ A **GPU with sufficient memory** (e.g., **NVIDIA RTX 3090**) is available to support 8B model inference.


📌 **Adding Handmade Custom Chunks**

Chunk files that respect this JSON format, can manually be added to [chunked_files](data/chunked_files) folder:
```json
{
    "id": {                 <--- A unique identifier (mandatory)
        "chunks": [],       <--- A list of text chunks (mandatory) 
        "metadata": "...",  <--- Additional metadata (optional) 
        ...                 <--- Other optional metadata fields
    },
    ...                     <--- Other dict of chunks and metadata
}
```
An example can be found in [hand_made_chunks.json](data/chunked_files/hand_made_chunks.json).

🚨 **Dealing with undesirable questions and prompts**

If a chunk file named [garbage_model.json](data/chunked_files/garbage_model.json) is included, its chunks will be treated as part of a **garbage model**. These chunks will be retrieved but **won't be passed to the LLM** but will instead follow an out-of-domain answer procedure, which helps deal with bad inputs. 

❌ **Dealing with censored words**

Some words are **censored** by our RAG, meaning the RAG **won't retrieve any chunk** if they appear in the query. The censored word list can be found in the [utils.py](src/rag_database_generator/utils.py) file.

### 2. Generate RAG Database

To compute the embeddings and generate the database: 
```bash
embed-files path/file1.json path/file2.json ...
```

> Use the `--help` flag to see available options for `generate-db`.

---

### Full pipeline

All previous steps (0, 1, and 2) can be executed in a single command, as shown below:
```bash
generate-db path/file1.pdf path/file2.pdf ...
```

## Test Your Database

To test the retrieval process, which identifies the most relevant chunks in the `rag_database.pkl` file, run:
```bash
retrieve --rag-database data/databases/medical_db.pkl
```

> Use the `--help` flag to see available options for `rag`.
> 
> **Note:** Some words are censored by our RAG, meaning the system will not respond if they appear in the query. The censored word list can be found in the [utils.py](src/rag_database_generator/utils.py) file.

---

## 📝 Jupyter Notebooks (Optional GUI Alternative)

If you prefer working in a Jupyter Notebook environment instead of the command line, two example notebooks are available:

- [`generate_retrieval_database.ipynb`](notebooks/generate_retrieval_database.ipynb) — A step-by-step example to generate your RAG database (PDF parsing, chunking, and embeddings) from the [Medical.pdf](notebooks/Medical.pdf) file.
- [`run_llm_with_rag.ipynb`](notebooks/run_llm_with_rag.ipynb) — An interactive notebook to test chunk retrieval and answer generation with any Hugging Face LLM.

> ⚠️ You still need to meet all environment and model requirements (e.g., GPU, Hugging Face access) when using these notebooks. And the data folder is `example_notebooks/`.

> **Note:** The Jupyter dependencies are **optional** and are not installed by the default `./run.sh --install`. Install them explicitly before running the notebooks:

```bash
# Install the optional notebook dependencies (jupyter, nbmake)
./run.sh --install-notebooks
```

To launch the notebooks, run:
```bash
./run.sh --notebook
```
---

<br>
<p align="center">
  <a href="https://www.nxp.com">
    <img src="https://mcuxpresso.nxp.com/static/icon/nxp-logo-color.svg" width="100"/>
  </a>
</p>