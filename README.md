# Hindsight

An unofficial implementation of 
[Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](http://arxiv.org/abs/2512.12818) 
in Rust.
This project implements the agentic memory management architecture proposed in the paper.


## Requirements

- Rust 1.85+
- Docker (for PostgreSQL)
- OpenAI-compatible LLM endpoint (e.g., LM Studio, Ollama, vLLM)

## Run

1. Start PostgreSQL with pgvector:

```bash
docker compose up -d
```

2. Set environment variables (defaults shown — only override if needed):

```bash
export DATABASE_URL="postgres://hindsight:hindsight@localhost:5432/hindsight"
export LLM_BASE_URL="http://127.0.0.1:1234"
export LLM_API_KEY="local"
export CHAT_MODEL="google/gemma-3-27b"
export EMBED_MODEL="nomic-ai/nomic-embed-text-v1.5-GGUF"
export EMBEDDING_DIM=768
```

3. Build and run:

```bash
cargo run
```
