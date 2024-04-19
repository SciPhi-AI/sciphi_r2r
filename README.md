<p align="left">
  <a href="https://r2r-docs.sciphi.ai"><img src="https://img.shields.io/badge/docs.sciphi.ai-3F16E4" alt="Docs"></a>
  <a href="https://discord.gg/p6KqD2kjtB"><img src="https://img.shields.io/discord/1120774652915105934?style=social&logo=discord" alt="Discord"></a>
  <a href="https://github.com/SciPhi-AI"><img src="https://img.shields.io/github/stars/SciPhi-AI/R2R" alt="Github Stars"></a>
  <a href="https://github.com/SciPhi-AI/R2R/pulse"><img src="https://img.shields.io/github/commit-activity/w/SciPhi-AI/R2R" alt="Commits-per-week"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License: MIT"></a>
</p>

<img src="./docs/pages/r2r.png" alt="Sciphi Framework">
<h3 align="center">
Build, deploy, and optimize your RAG system.
</h3>

## About

R2R (RAG to Riches) offers a fast and efficient framework for serving high-quality Retrieval-Augmented Generation (RAG) to end users. The framework is designed with customizable pipelines and a feature-rich FastAPI implementation, enabling developers to quickly deploy and scale RAG-based applications.


## Why?

R2R was conceived to bridge the gap between local LLM experimentation and scalable production solutions. **R2R is to LangChain/LlamaIndex what NextJS is to React**. A JavaScript client for R2R deployments can be [found here](https://github.com/SciPhi-AI/r2r-js).

### Key Features

- **🚀 Deploy**: Instantly launch production-ready RAG pipelines with streaming capabilities.
- **🧩 Customize**: Tailor your pipeline with intuitive configuration files.
- **🔌 Extend**: Enhance your pipeline with custom code integrations.
- **⚖️ Autoscale**: Scale your pipeline effortlessly in the cloud using [SciPhi](https://sciphi.ai/).
- **🤖 OSS**: Benefit from a framework developed by the open-source community, designed to simplify RAG deployment.

## Demo(s)

Using the cloud application to deploy the pre-built basic pipeline:

https://www.loom.com/share/e3b934b554484787b005702ced650ac9

Note - the example above uses [SciPhi Cloud](https://sciphi.ai) to pair with the R2R framework for deployment and observability. SciPhi is working to launch a self-hosted version of their cloud platform as R2R matures.

## Links

[Join the Discord server](https://discord.gg/p6KqD2kjtB)

[R2R Docs Quickstart](https://r2r-docs.sciphi.ai/getting-started/quick-install)

[SciPhi Cloud Docs](https://docs.sciphi.ai/)

[Local RAG Tutorial](https://r2r-docs.sciphi.ai/tutorials/local_rag)

## Quick Install:

```bash
# use the `'r2r[all]'` to download all required deps
pip install 'r2r[eval]'

# setup env 
export OPENAI_API_KEY=sk-...
# Set `LOCAL_DB_PATH` for local testing
export LOCAL_DB_PATH=local.sqlite # robust providers available (e.g. qdrant, pgvector, ..)

# OR do `vim .env.example && cp .env.example .env`
# INCLUDE secrets and modify config.json
# if using cloud providers (e.g. pgvector, qdrant, ...)
```

## Docker:

```bash
docker pull emrgntcmplxty/r2r:latest

# Choose from CONFIG_OPTION in {`default`, `local_ollama`}
# For cloud deployment, select `default` and pass `--env-file .env`
# For local deployment, select `local_ollama`
docker run -d --name r2r_container -p 8000:8000 -e CONFIG_OPTION=local_ollama  emrgntcmplxty/r2r:latest
```

## Basic Example

[`Configurable Pipeline`](r2r/examples/servers/config_pipeline.py): Execute this script to select and serve a **Q&A RAG**, **Web RAG**, or **Agent RAG** pipeline. This starter pipeline supports ingestion, embedding, and question and the specified RAG, all accessible via a REST API.
   ```bash
   # launch the server
   # For ex., do `export CONFIG_OPTION=local_ollama` or ``--config=local_ollama` to run fully locally
   # For ex., do `export PIPELINE_OPTION=web` or ``--pipeline=web` to run WebRAG pipeline
   python -m r2r.examples.servers.config_pipeline --config=default --pipeline=qna
   ```

[`Question & Answer Example Client`](r2r/examples/clients/run_qna_client.py): This **client script** should be executed subsequent to the server startup above with `pipeline=qna` specified. It facilitates the upload of text entries and PDFs to the server using the Python client and demonstrates the management of document and user-level vectors through its built-in features.

   ```bash
   # run the client
   python -m r2r.examples.clients.run_qna_client ingest
   python -m r2r.examples.clients.run_qna_client search --query="What is the meaning of life?"
   ```
### Running Local RAG

[Refer here](https://r2r-docs.sciphi.ai/tutorials/local_rag) for a tutorial on how to modify the commands above to use local providers.

## Synthetic Queries Example

[`synthetic_query_pipeline.py`](r2r/examples/servers/synthetic_query_pipeline.py): Execute this script to start a backend server equipped with an advanced pipeline. This pipeline is designed to create synthetic queries, enhancing the RAG system's learning and performance.

   ```bash
   # launch the server
   python -m r2r.examples.servers.synthetic_query_pipeline
   ```

[`run_synthetic_query_client.py`](r2r/examples/clients/run_synthetic_query_client.py): Use this client script after the synthetic query pipeline is running. It's tailored for use with the synthetic query pipeline, demonstrating the improved features of the RAG system.

   ```bash
   # run the client
   python -m r2r.examples.clients.run_synthetic_query_client
   ```

## Extra Examples

[`reducto_pipeline.py`](r2r/examples/servers/reducto_pipeline.py): Launch this script to activate a backend server that integrates a Reducto adapter for enhanced PDF ingestion.

   ```bash
   # launch the server
   python -m r2r.examples.servers.reducto_pipeline
   ```

## Core Abstractions

The framework primarily revolves around three core abstractions:

- The **Ingestion Pipeline**: Facilitates the preparation of embeddable 'Documents' from various data formats (json, txt, pdf, html, etc.). The abstraction can be found in [`ingestion.py`](r2r/core/pipelines/ingestion.py) and relevant documentation is available [here](https://r2r-docs.sciphi.ai/deep-dive/ingestion).

- The **Embedding Pipeline**: Manages the transformation of text into stored vector embeddings, interacting with embedding and vector database providers through a series of steps (e.g., extract_text, transform_text, chunk_text, embed_chunks, etc.). The abstraction can be found in [`embedding.py`](r2r/core/pipelines/embedding.py) and relevant documentation is available [here](https://r2r-docs.sciphi.ai/deep-dive/embedding).

- The **RAG Pipeline**: Works similarly to the embedding pipeline but incorporates an LLM provider to produce text completions. The abstraction can be found in [`rag.py`](r2r/core/pipelines/rag.py) and relevant documentation is available [here](https://r2r-docs.sciphi.ai/deep-dive/rag).

- The **Eval Pipeline**: Samples some subset of rag_completion calls for evaluation. Currently [DeepEval](https://github.com/confident-ai/deepeval) and [Parea](https://github.com/parea-ai/parea-sdk-py) are supported. The abstraction can be found in [`eval.py`](r2r/core/pipelines/eval.py) and relevant documentation is available [here](https://r2r-docs.sciphi.ai/deep-dive/eval).

Each pipeline incorporates a logging database for operation tracking and observability.
