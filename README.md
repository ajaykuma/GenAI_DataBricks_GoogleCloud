# GenAI_DataBricks_GoogleCloud

My exploratory work on AI/GenAI — Google AI/APIs, Databricks Platform, HuggingFace/Transformers, OpenAI/Azure AI, and integration with distributed search platforms such as ELK, OpenSearch, and vector stores (DBs).

> 🚧 This repository is constantly being refined and improved. Stay tuned!

## What's Inside

| # | Folder | Contents |
|---|--------|----------|
| 0 | [Documentation-Books](./0-Documentation-Books) | Notes / books / reference material |
| 1 | [Setup_options](./1-Setup_options) | Environment & setup options for the examples below |
| 2 | [General](./2-General) | General-purpose Notes |
| 3 | [Datasets](./3-Datasets) | Sample/reference datasets used/synthesized across the projects here |
| 4 | [ChatGpt_Interactions](./4-ChatGpt_Interactions) | Raw GPT/LLM interaction samples — see note below |
| 5 | [Hugging_Face_n_OpenAI_Azure](./5-Hugging_Face_n_OpenAI_Azure) | HuggingFace / OpenAI / Other LLMs /locally deployed or on Azure/GCP (Frameworks explored: LangChain, LangGraph, AutoGen, CrewAI) |
| 6 | [GCP](./6-GCP) | Code examples/notes on Google Cloud & Vertex AI |
| 7 | [Databricks](./7-Databricks) | GenAI work on the Databricks platform |
| 8 | [Real-world-projects-xmpls](./8-Real-world-projects-xmpls) | Real-world project examples (shared where no NDA conflicts)

> 📌 **Note on `4-ChatGpt_Interactions`:** This folder contains raw sample interaction logs only. Refined, structured guides are maintained separately and available on request — email **ajaykuma24@gmail.com**.

> 🚧 When browsing folders, look for `*.txt` files for instructions/explanations, and use `*.py`/`*.ipynb` files from `Sample_Codes`.
> Remember to install packages (in your main or a virtual env) to test code examples, or use Colab notebooks from Google Cloud.

## Models Explored & Used
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2
- OpenAI/gpt-3.5-turbo
- OpenAI/gpt-4.1 (also 5.1)
- OpenAI/clip-vit-base-patch32
- mistralai/Mistral-7B-Instruct-v0.1
- tiiuae/falcon-7b-instruct
- runwayml/stable-diffusion-v1-5
- dreamlike-art/dreamlike-photoreal-2.0
- google/flan-t5-base (or -large, -XL, -XXL)
- google/gemini-1.5-pro
- google/gemini-2.0-flash-001
- google/gemini-1.5-flash
- google/vit-base-patch16-224
- google/imagen-alpha
- bert-base-multilingual-uncased-sentiment
- bert-base-uncased
- glove-wiki-gigaword-50
- text-embedding-ada-002   # most common, older
- text-embedding-3-small   # newer, cheaper
- text-embedding-3-large   # newest, most accurate
- *more to be added*

## License & Commercial Use

This repository is licensed under the [MIT License](./LICENSE) — free to use, fork, and build on.

📩 Extended, production-grade versions of these projects (additional pipelines, monitoring, scaling, and integration work) are maintained separately and available for commercial licensing or collaboration. Reach out via GitHub if interested, or email **ajaykuma24@gmail.com**.
