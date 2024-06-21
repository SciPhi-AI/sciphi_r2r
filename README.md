<p align="left">
  <a href="https://r2r-docs.sciphi.ai"><img src="https://img.shields.io/badge/docs.sciphi.ai-3F16E4" alt="Docs"></a>
  <a href="https://discord.gg/p6KqD2kjtB"><img src="https://img.shields.io/discord/1120774652915105934?style=social&logo=discord" alt="Discord"></a>
  <a href="https://github.com/SciPhi-AI"><img src="https://img.shields.io/github/stars/SciPhi-AI/R2R" alt="Github Stars"></a>
  <a href="https://github.com/SciPhi-AI/R2R/pulse"><img src="https://img.shields.io/github/commit-activity/w/SciPhi-AI/R2R" alt="Commits-per-week"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License: MIT"></a>
</p>

<img src="./docs/pages/r2r.png" alt="R2R Answer Engine">
<h3 align="center">
The ultimate open source AI powered answer engine
</h3>

# About

R2R (RAG to Riches)  was designed to bridge the gap between local LLM experimentation and scalable, production-ready Retrieval-Augmented Generation (RAG). R2R provides a comprehensive and SOTA RAG system for developers, built around a [RESTful API](/api-reference/introduction) for ease of use.

For a more complete view of R2R, check out our [documentation](https://r2r-docs.sciphi.ai/).

## Key Features
- **📁 Multimodal Support**: Ingest files ranging from `.txt`, `.pdf`, `.json` to `.png`, `.mp3`, and more.
- **🔍 Hybrid Search**: Combine semantic and keyword search with reciprocal rank fusion for enhanced relevancy.
- **🔗 Graph RAG**: Automatically extract relationships and build knowledge graphs.
- **🗂️ App Management**: Efficiently manage documents and users with rich observability and analytics.
- **🌐 Client-Server**: RESTful API support out of the box.
- **🧩 Configurable**: Provision your application using intuitive configuration files.
- **🔌 Extensible**: Develop your application further with easy builder + factory pattern.
- **🖥️ Dashboard**: Use the [R2R Dashboard](https://github.com/SciPhi-AI/R2R-Dashboard), an open-source React+Next.js app for a user-friendly interaction with your pipelines.

## Table of Contents
1. [Install](#install)
2. [R2R Quickstart](#r2r-quickstart)
3. [R2R Dashboard](#r2r-dashboard)
4. [Community and Support](#community-and-support)
5. [Contributing](#contributing)

# Install

<details open>
<summary><b>Installing with Pip</b>&nbsp;🐍 </summary>

```bash
pip install r2r

# setup env
export OPENAI_API_KEY=sk-...
export POSTGRES_USER=YOUR_POSTGRES_USER
export POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD
export POSTGRES_HOST=YOUR_POSTGRES_HOST
export POSTGRES_PORT=YOUR_POSTGRES_PORT
export POSTGRES_DBNAME=YOUR_POSTGRES_DBNAME
```
</details>

<details>
<summary><b>Installing with Docker</b>&nbsp;🐳</summary>

Note: The R2R client must still be installed, even when running with Docker. Download the Python client with `pip install r2r`.

To run R2R using Docker:

```bash
docker pull emrgntcmplxty/r2r:latest

docker run -d \
  --name r2r \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  emrgntcmplxty/r2r:latest
```

For local LLMs:

```bash
docker run -d \
  --name r2r \
  --add-host=host.docker.internal:host-gateway \
  -p 8000:8000 \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434 \
  -e CONFIG_OPTION=local_ollama \
  emrgntcmplxty/r2r:latest
```
</details>

For full installation details and environment setup, please refer to our [Installation Guide](https://r2r-docs.sciphi.ai/installation).

# R2R Quickstart

The [R2R Quickstart](https://r2r-docs.sciphi.ai/quickstart) provides a step-by-step guide to get up and running with R2R in minutes. Here's a brief overview:

<details open>
<summary><b>Document Ingestion and Management</b></summary>

1. **Ingest Files**:
   ```bash
   python -m r2r.examples.quickstart ingest_as_files
   ```

2. **View Document Info**:
   ```bash
   python -m r2r.examples.quickstart documents_overview
   ```

3. **View User Overview**:
   ```bash
   python -m r2r.examples.quickstart users_overview
   ```
</details>

<details>
<summary><b>Search and RAG Operations</b></summary>

1. **Search Documents**:
   ```bash
   python -m r2r.examples.quickstart search --query="Who was Aristotle?"
   ```

2. **RAG Completion**:
   ```bash
   python -m r2r.examples.quickstart rag --query="What was Uber's profit in 2020?"
   ```

3. **Streaming RAG**:
   ```bash
   python -m r2r.examples.quickstart rag --query="What was Lyft's profit in 2020?" --streaming=true
   ```

4. **Hybrid Search RAG**:
   ```bash
   python -m r2r.examples.quickstart rag --query="Who is John Snow?" --do_hybrid_search
   ```
</details>

For more detailed examples and advanced features, please refer to our [Quickstart Guide](https://r2r-docs.sciphi.ai/quickstart).

# R2R Dashboard

Interact with R2R using our [open-source React+Next.js dashboard](https://github.com/SciPhi-AI/R2R-Dashboard). Check out the [Dashboard Cookbook](https://r2r-docs.sciphi.ai/cookbooks/dashboard) to get started!

# Community and Support

- [Discord](https://discord.gg/p6KqD2kjtB): Chat live with maintainers and community members
- [Github Issues](https://github.com/SciPhi-AI/R2R/issues): Report bugs and request features

Explore our [R2R Docs](https://r2r-docs.sciphi.ai/) for tutorials and cookbooks on various R2R features and integrations, including:
- [Client-Server](/cookbooks/client-server)
- [Local RAG](/cookbooks/local-rag)
- [Hybrid Search](/cookbooks/hybrid-search)
- [Reranking](/cookbooks/rerank-search)
- [GraphRAG](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph)

# Contributing

We welcome contributions of all sizes! Here's how you can help:

- Open a PR for new features, improvements, or better documentation.
- Submit a [feature request](https://github.com/SciPhi-AI/R2R/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=) or [bug report](https://github.com/SciPhi-AI/R2R/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=)

### Our Contributors
<a href="https://github.com/SciPhi-AI/R2R/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SciPhi-AI/R2R" />
</a>
