# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based web application that fetches and summarizes arXiv academic papers using AI summarization. The entire application is a single-file Python script (`arxiv_summarizer.py`).

## Architecture

**Single-File Application**: The entire app exists in `arxiv_summarizer.py` with three main components:

1. **arXiv API Integration** (`fetch_arxiv_results()`): Fetches papers from arXiv API using urllib, returns XML responses
2. **AI Summarization** (`summarize_text()`, `load_summarization_model()`): Uses HuggingFace Transformers (facebook/bart-large-cnn model) to generate summaries. Model is cached using `@st.cache_resource` to avoid reloading
3. **Streamlit UI** (`parse_arxiv_to_polars_and_summarize()`): Provides search interface, displays results in Polars DataFrame, shows both individual article summaries and an overall summary paragraph

**Data Flow**:
- User enters search term + count → arXiv API query → XML parsing → individual article summarization → Polars DataFrame creation → overall summary generation → display

## Common Commands

### Development
```bash
# Activate virtual environment (required before running)
source .venv/bin/activate

# Run the Streamlit application
streamlit run arxiv_summarizer.py

# Install/update dependencies
uv pip install -e .
```

### Dependencies
Project uses uv for package management. Key dependencies:
- `streamlit`: Web UI framework
- `transformers`: HuggingFace models for summarization (facebook/bart-large-cnn)
- `torch`: Required backend for transformers
- `polars`: DataFrame library for handling search results
- `matplotlib`: Included but not currently used in the app

## Key Implementation Details

**Namespace Handling**: arXiv API returns Atom XML format with namespace `http://www.w3.org/2005/Atom`. All XML parsing must use the `namespaces` dict with `atom:` prefix.

**Summarization Strategy**:
- Individual summaries: max_length=140, min_length=30
- Overall summary: Concatenates all article summaries, then summarizes the concatenation
- Model cached at module level to prevent reloading on each Streamlit rerun

**Custom Styling**: Dark background (#3a3c40) applied via inline CSS in `background_color_css` variable.

## Python Version

Requires Python >=3.10 (specified in pyproject.toml)
