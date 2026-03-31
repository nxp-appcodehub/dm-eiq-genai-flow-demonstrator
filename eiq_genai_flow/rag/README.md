![](assets/RAG.png)

# Retrieval-Augmented Generation (RAG)


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

---
## Overview
<div style="text-align: justify;">

**Retrieval-Augmented Generation (RAG)** enhances the knowledge base of a Large Language Model (LLM) for a target domain, device, or context. RAG helps control the LLM’s behavior, mitigate hallucinations and customize responses based on examples. RAG involves injecting additional contextual information into the input prompt of the LLM, retrieved from a compact knowledge database stored on the edge device. 

1. **Database Generation**: A knowledge database is created offline, consisting of pairs of chunks (pieces of text) and their corresponding embeddings (vectorial representations from an embedding model).
2. **Retrieval**: Each query is encoded into an embedding using the same model. The query embedding is then compared to the database embeddings and the most similar to the query are added to the prompt.
3. **Generation**: A LLM receives the formated prompt and generates an answer.
</div>

---
## Features

### Database generator (PC only)
* 📚 Advanced PDF document parsing leveraging [Docling](https://github.com/DS4SD/docling "Go to Docling repository").
* ✂️ Solutions to break down large textual data using chunking algorithms. Including [HiRAG](https://openreview.net/forum?id=cWWb9cgSVi "HiRAG paper") that offers an advanced chunk reformatting.
* 🧩 Embedding model adapted for NXP platforms to generate RAG databases.

### Inference (PC & i.MX)
* 🔍 A retrieval engine that finds the most relevant chunks for a given query.
---

## Table of Contents

[Installation](#installation)

[Configurable parameters](#configurable-parameters)

[Custom Database Generation](#custom-database-generation-pc-only)

  * [(Optional) Parse PDF files](#optional-parse-pdf-files-pc-only)
  * [Generate Chunks](#generate-chunks-pc-only)
  * [Generate RAG Database](#generate-rag-database-pc-only)

[Classify Inputs](#classify-inputs)

[Custom Database Testing](#custom-database-testing-pc--imx)

[Support](#support)

[Release Notes](#release-notes)

---
<a name="installation"></a>
## Installation

<a name="set-up-the-environment"></a>
### Set up the environment:

**Works only on Linux environment. This package has been tested thoroughly with [Python 3.13.2](https://www.python.org/downloads/).**

Install Runtime & Development Dependencies (Dev):
```bash
cd rag
# Install Runtime Dependencies (i.MX & PC):
pip install -e .
# Install Runtime and Development Dependencies (PC):
pip install -e .[dev]
```

<a name="optional-setting-up-a-hugging-face-environment-"></a>
#### (Optional) Setting Up a Hugging Face Environment: 

1️⃣ [Create a Hugging Face account](https://huggingface.co/join).<br>
2️⃣ [Generate a personal read access token](https://huggingface.co/settings/tokens).<br>
3️⃣ [Log in to Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/en/guides/cli):
   ```bash
   # Replace $HF_TOKEN with the generated personal access token
   huggingface-cli login --token $HF_TOKEN  
   ```

---
<a name="configurable-parameters"></a>
## Configurable parameters

Various parameters in the [config.py](src/rag/config.py) file are available to customize the behavior of the project.  

| Parameter                   | Type    | Default                                      | Description                                                                                                                                       |
|-----------------------------|---------|----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------| 
| `out_of_domain_source_list` | `tuple` | `("censored_queries", "garbage_model.json")` | Specifies the list of sources that must be considered as out of domain for the `QueryClassifier`                                                  |
| `top_k`                     | `int`   | `3`                                          | Specifies the number of top similar chunks to retrieve from `rag_database.pkl` based on their similarity to the user's query.                     |
| `reranking`                 | `bool`  | `True`                                       | Determines whether to rerank the `top_k` chunks to further improve relevance before selecting the `best_k` elements.                              |
| `best_k`                    | `int`   | `1`                                          | Specifies the number of the most relevant chunks selected and returned by the system.                                                             |
| `chunk_size`                | `int`   | `128`                                        | Defines the length (in characters) of each text chunk. Larger chunks may improve response quality but increase processing time.                   |
| `chunk_overlap`             | `int`   | `64`                                         | Defines the overlap between consecutive chunks, typically set to half the `chunk_size` to preserve context across chunks.                         |
| `database_description`      | `str`   | `"..."`                                      | Database description that will be embedded in the RAG database during its generation.                                                             |

---

<a name="custom-database-generation-pc-only"></a>
## Custom Database Generation (PC only)

These are the steps for creating a RAG database.

💡 To understand the workflow and its features, example files in the [data folder](src/data) are included. These files illustrate the entire process, from [parsing](src/data/parsed_files) and [chunking](src/data/chunked_files) the [Medical.pdf](src/data/input_files/Medical.pdf) to incorporating handcrafted chunks. The resulting database is then used in the `eIQ GenAI Flow Demonstrator`.

<a name="optional-parse-pdf-files-pc-only"></a>
### (Optional) Parse PDF Files (PC only)

To parse PDFs into Markdown files, we use [Docling](https://github.com/docling-project/docling/tree/main), so the [Dev environment](#set-up-the-environment) is needed. FlashAttention is also needed, to install run:
```bash
pip install flash-attn --no-build-isolation
```

> **Note:** [Set up a Hugging Face environment](#optional-setting-up-a-hugging-face-environment-) may be needed before running Docling.
> 
> If any issues occur, please refer to the [Docling documentation](https://docling-project.github.io/docling/).

📂 **Where to Place PDF files**

Move the PDF files into the `input_files` folder:
```
.
├── assets
├── example_notebooks
├── Licence
├── pyproject.toml
├── README.md
├── install.sh
├── LICENSE.txt
└── src
    ├── data
    │    ├── chunked_files
    │    ├── input_files
    │    │        └── 📄 <<< PUT PDF FILES HERE >>>
    │    ├── parsed_files
    │    └── rag_database.pkl
    ├── document_parsing
    ├── rag
    └── shared_utils
```

To process PDFs and extract text, run:
```bash
python -m document_parsing
```
> ⚠️ **PDF with tables are not yet supported**
>
> Use the `--help` flag to see available options for `document_parsing`.

<a name="generate-chunks-pc-only"></a>
### Generate Chunks (PC only)

To generate text chunks from Markdown files: 
```bash
python -m rag.preprocessing.generate_chunks
```
> Use the ``--help`` flag to see available options for `generate_chunks`.

> By default, the system uses the [HiRAG](https://openreview.net/forum?id=cWWb9cgSVi "HiRAG paper") chunking method, which requires running an LLM. 
> 
> To use **HiRAG**, ensure that:
>
> ✅ Access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) was requested.<br>
> ✅ [A Hugging Face environment is set up](#optional-setting-up-a-hugging-face-environment-).<br>
> ✅ A **GPU with sufficient memory** (e.g., **NVIDIA RTX 3090**) is available to support 8B model inference.


📌 **Adding Custom Chunks**

Chunk files that respect this JSON format, can manually be added to [chunked_files](src/data/chunked_files):
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
An example can be found in [Medical_hand_made_chunks.json](src/data/chunked_files/Medical_hand_made_chunks.json).

<a name="generate-rag-database-pc-only"></a>
### Generate RAG Database (PC only)

To compute the embeddings and generate the database, first set the right `database description` in [config.py](src/rag/config.py), then run: 
```bash
python -m rag.preprocessing.generate_embeddings
```

> Use the `--help` flag to see available options for `generate_embeddings`.

---
<a name="classify-inputs"></a>
## Classify Inputs

The **`QueryClassifier`** class categorizes each incoming query into one of several classes based on metadata and retrieval results. (See the implementation in [`retrieval.py`](src/rag/retrieval.py)).

| Query Category        | Criteria                                                                                    | Purpose                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **`CENSORED`**  | The query contains a term from the [censored list](./src/rag/utils.py).                     | Prevents unsafe or undesired queries from being processed by the LLM.                            |
| **`INTENT`**    | The top-ranked chunk contains an `intent` field in its metadata.                            | Enables command-level actions (e.g., *“Show me my glucose level.”*) to be detected and executed. |
| **`REJECTED`**  | Over half of the retrieved chunks originate from `garbage_model.json` (out-of-domain data). | Filters out questions unrelated to the application domain. (e.g. [garbage_model.json](./src/data/chunked_files/garbage_model.json))                                      |
| **`AMBIGUOUS`** | The mean similarity score of all retrieved chunks falls below the confidence threshold.     | Signals that retrieved content does not strongly match the query (low confidence).               |
| **`ACCEPTED`**  | Default case — none of the above conditions apply.                                          | Indicates that the LLM can be safely and confidently prompted.                                   |

---
<a name="custom-database-testing-pc--imx"></a>
## Custom Database Testing (PC & i.MX)

To test the retrieval process, which identifies the most relevant chunks in the `rag_database.pkl` file, on PC and i.MX run:

```bash
# On the medical example
python -m rag
```

> Use the `--help` flag to see available options for `rag`.

If the retrieved chunks are satisfiying:

-  To prompt a HuggingFace LLM with `RAG`, run:

```bash
# On the medical example
python -m rag.run_llm
```


- To test `RAG` and the `QueryClassifier` with `eIQ GenAI Flow`, run in project root:
```bash
# i.MX only
python eiq_genai_flow.py -r
```
---

## 📝 Jupyter Notebooks (Optional GUI Alternative)

If you prefer working in a Jupyter Notebook environment instead of the command line, two example notebooks are available:

- [`generate_retrieval_database.ipynb`](example_notebooks/generate_retrieval_database.ipynb) — A step-by-step example to generate your RAG database (PDF parsing, chunking, and embeddings) from the [Medical.pdf](notebooks/Medical.pdf) file.
- [`run_llm_with_rag.ipynb`](example_notebooks/run_llm_with_rag.ipynb) — An interactive notebook to test chunk retrieval and answer generation with supported Hugging Face LLM.

> ⚠️ You still need to meet all environment and model requirements (e.g., GPU, Hugging Face access) when using these notebooks.

To launch the notebooks, run:
```bash
jupyter notebook
```
---

<a name="support"></a>
## Support

For more general technical questions, use the [NXP Community Forum](https://community.nxp.com/).

---
<a name="release-notes"></a>
## Release Notes
| Version | Description / Update                     | Date                         |
|---------|------------------------------------------|------------------------------|
| 2.0     | Initial release on Application Code Hub. | November 21<sup>th</sup> 2025 |

<br>
<p align="center">
  <a href="https://www.nxp.com">
    <img src="https://mcuxpresso.nxp.com/static/icon/nxp-logo-color.svg" width="100"/>
  </a>
</p>