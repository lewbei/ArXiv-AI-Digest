# 📚 ArXiv-AI-Digest

> **AI-powered ArXiv paper summarization using Gemini CLI for comprehensive research analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

**ArXiv-AI-Digest** is an automated research tool that downloads, extracts, and summarizes academic papers from ArXiv using Google's Gemini CLI. Transform lengthy research papers into comprehensive, structured analysis in minutes.

### ✨ Key Features

- **📥 ArXiv Paper Download**: Automated downloading from ArXiv by search query and date range
- **📄 PDF Text Extraction**: Robust text extraction using PyPDF2 and pdfplumber
- **🧠 AI-Powered Summarization**: Comprehensive research analysis using Gemini 2.5 Flash
- **📝 Research-Grade Analysis**: 10-section analysis framework covering methodology, results, limitations, and impact
- **📊 Batch Processing**: Process multiple papers automatically
- **💾 Markdown Output**: Clean, structured summaries in markdown format

## 🚀 Quick Start

### Check Examples First! 👀

Before starting, check the `example/` folder to see:
- **Sample PDFs**: Real ArXiv papers (`2001.00127v2.pdf`, etc.)
- **Generated Summaries**: AI-generated analysis in markdown format
- **Expected Output**: What your summaries will look like

### Prerequisites

1. **Python Environment**: Miniconda with deep learning environment
2. **Gemini CLI**: Google Gemini CLI installed via npm
3. **PDF Libraries**: PyPDF2 or pdfplumber for text extraction

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Gemini CLI (requires Node.js)
npm install -g @google/gemini-cli
gemini auth login
```

### Basic Usage

```bash
# 1. Download papers from ArXiv
python download_arxiv_papers.py

# 2. Summarize all downloaded papers
python summarize_papers.py

# 3. Test with single paper
python test_single_pdf.py
```

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   ArXiv Downloader  │────│  PDF Text Extractor  │────│  Gemini Summarizer  │
│                     │    │                      │    │                     │
│ • Query ArXiv API   │    │ • PyPDF2 extraction  │    │ • Gemini CLI client │
│ • Download PDFs     │    │ • Text preprocessing │    │ • Research analysis │
│ • Save metadata     │    │ • Error handling     │    │ • Markdown output   │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### File Structure

```
├── download_arxiv_papers.py    # ArXiv paper downloader
├── extract_pdf_text.py         # PDF text extraction utility
├── summarize_papers.py         # Main summarization engine
├── best_prompt.txt             # Research analysis prompt template
├── example/                    # Example PDFs and generated summaries
│   ├── 2001.00127v2.pdf       # Sample ArXiv paper
│   ├── 2001.00127v2.md        # Generated summary example
│   └── ...                    # Additional examples
├── arxiv_papers/               # Downloaded PDF storage
├── summaries/                  # Generated markdown summaries
└── src/services/               # Minimal AI model integration
    ├── model_interface.py      # Abstract model interfaces
    └── cli_clients.py          # Gemini CLI implementation
```

## 📥 Downloading Papers

### Configure Search Parameters

Edit `download_arxiv_papers.py`:

```python
# Customize these parameters
search_term = "transformer neural networks"  # Your search query
max_total_results = 100                      # Number of papers to download
year = 2023                                  # Year to search
```

### Run Download

```bash
python download_arxiv_papers.py
```

**Output:**
- PDFs saved to `arxiv_papers/` directory
- Metadata saved to `arxiv_papers/paper_metadata.json`
- Progress and statistics displayed

## 📄 Text Extraction

### Standalone Usage

```bash
# Extract text from single PDF
python extract_pdf_text.py paper.pdf
```

### Supported Libraries
- **PyPDF2**: Primary extraction method
- **pdfplumber**: Fallback for complex PDFs
- **Automatic fallback**: If one fails, tries the other

## 🤖 AI Summarization

### Research Analysis Framework

The system uses a comprehensive 10-section analysis framework:

1. **Research Context & Motivation**
2. **Methodology & Approach** 
3. **Technical Contributions**
4. **Results & Findings**
5. **Comparative Analysis**
6. **Limitations & Critical Assessment**
7. **Impact & Significance**
8. **Implementation & Adoption**
9. **Future Work & Recommendations**
10. **Research Integrity & Quality**

### Gemini CLI Configuration

Uses **Gemini 2.5 Flash** for:
- ⚡ Fast processing (cost-effective)
- 🎯 High-quality research analysis
- 📝 1000-1500 word comprehensive summaries

### Run Summarization

```bash
# Process all PDFs in arxiv_papers/
python summarize_papers.py
```

**Output:**
- Markdown files in `summaries/` directory
- Each summary includes paper title, publication date, and comprehensive analysis
- Progress tracking and error handling

## 📋 Configuration

### Environment Setup

```bash
# Python environment (WSL/Linux)
/mnt/c/Users/lewka/miniconda3/envs/deep_learning/python.exe

# Gemini CLI requires Node.js environment
source ~/.nvm/nvm.sh
gemini --help  # Verify installation
```

### Prompt Customization

Edit `best_prompt.txt` to customize the analysis framework:
- Modify analysis sections
- Adjust word count requirements
- Change focus areas (technical vs. business)

## 🔧 Advanced Usage

### Custom Search Queries

```python
# In download_arxiv_papers.py
search_queries = [
    "attention mechanism transformer",
    "reinforcement learning deep Q",
    "computer vision CNN ResNet"
]
```

### Batch Processing Different Years

```python
for year in range(2020, 2024):
    download_arxiv_papers_api(search_term, max_results=50, year=year)
```

### Single Paper Processing

```python
from extract_pdf_text import extract_pdf_text

# Extract and process single paper
text = extract_pdf_text("paper.pdf")
# ... process with custom prompt
```

## 🛠️ Troubleshooting

### Common Issues

#### "Gemini CLI is not available"
```bash
# Check Gemini CLI installation
which gemini
gemini --help

# Verify Node.js environment
source ~/.nvm/nvm.sh
which node
```

#### PDF Extraction Fails
```bash
# Install alternative PDF library
pip install pdfplumber

# Try manual text extraction
python extract_pdf_text.py problem_paper.pdf
```

#### ArXiv API Rate Limiting
- Built-in 5-second delays between requests
- Reduce `max_total_results` if needed
- Check ArXiv API status

### Debug Scripts

```bash
# Test Gemini CLI integration
python debug_gemini.py

# Test environment setup
python test_bash.py
python test_nvm.py
```

## 📊 Example Output

### 🎯 **See Real Examples**

Check the `example/` folder for complete examples:
- **`example/2001.00127v2.pdf`** - Original ArXiv paper (Reinforcement Quantum Annealing)
- **`example/2001.00127v2.md`** - Complete AI-generated analysis
- **`example/2001.00234v1.pdf`** - Another sample paper
- **`example/2001.00234v1.md`** - Another detailed summary

### Downloaded Papers Structure
```
arxiv_papers/
├── 2001.00127v2.pdf
├── 2001.00234v1.pdf
├── paper_metadata.json
└── ...
```

### Generated Summary Format
```markdown
# Reinforcement Quantum Annealing: A Quantum-Assisted Learning Automata Approach

**Published Date:** 2020-01-01T16:41:58Z

---

## RESEARCH CONTEXT & MOTIVATION

This research addresses the fundamental challenge of combining quantum annealing...

## METHODOLOGY & APPROACH

The authors propose a novel framework that integrates reinforcement learning...

[... comprehensive 1500-word analysis with 10 detailed sections ...]
```

💡 **Tip**: Look at the example files to understand the quality and depth of analysis before running the system on your own papers!

## 🤝 Contributing

### Areas for Improvement
- **Additional PDF Extractors**: Support for more complex PDF formats
- **Multiple AI Models**: Integration with other AI providers
- **Advanced Filtering**: Better paper selection and filtering
- **Output Formats**: PDF, Word, or other export formats

### Development Setup

```bash
# Clone and setup
git clone <repository>
cd ArXiv-AI-Digest

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_single_pdf.py
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **ArXiv** for providing open access to research papers
- **Google** for Gemini CLI and API access
- **PyPDF2/pdfplumber** for PDF processing capabilities
- The **open source research community**

---

**ArXiv-AI-Digest** - Made for the research community 🎓

*Transform complex research into digestible insights with AI*