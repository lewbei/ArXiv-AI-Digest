import os
import sys
import asyncio
import json
from pathlib import Path

# Add the directory containing extract_pdf_text.py and services to the Python path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'src' / 'services'))

from extract_pdf_text import extract_pdf_text
from services.model_interface import ModelConfig, ModelRequest, ModelProvider
from services.cli_clients import GeminiCLIClient

async def get_gemini_summary_cli(prompt_text: str, model_id: str = "gemini-2.5-flash") -> str:
    """Sends the prompt to the Gemini CLI model and returns the summary."""
    config = ModelConfig(
        provider=ModelProvider.GEMINI_CLI,
        model_id=model_id,
        display_name=f"Gemini CLI {model_id}"
    )
    client = GeminiCLIClient(config)

    if not client.is_available():
        return "Error: Gemini CLI is not available or not configured correctly."

    request = ModelRequest(prompt=prompt_text)
    try:
        response = await client.generate(request)
        return response.content
    except Exception as e:
        print(f"Error generating content with Gemini CLI: {e}")
        return f"Error: Could not generate summary. {e}"

async def summarize_papers(pdf_dir: str, prompt_file: str, output_dir: str = "summaries"):
    """Summarizes PDF papers one by one using the Gemini CLI model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(prompt_file, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    metadata_file = os.path.join(pdf_dir, "paper_metadata.json")
    paper_metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            paper_metadata = json.load(f)
    else:
        print(f"Warning: {metadata_file} not found. Summaries will not include title/date.")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to summarize in {pdf_dir}.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        paper_id = os.path.splitext(pdf_file)[0]
        summary_filename = paper_id + ".md"
        summary_path = os.path.join(output_dir, summary_filename)

        if os.path.exists(summary_path):
            print(f"Summary for {pdf_file} already exists. Skipping.")
            continue

        print(f"\nProcessing {pdf_file}...")
        extracted_text = extract_pdf_text(pdf_path)

        if extracted_text:
            full_prompt = base_prompt + "\n\n" + extracted_text
            print("Sending text to Gemini CLI for summarization...")
            summary_content = await get_gemini_summary_cli(full_prompt)

            # Prepend title and date if available
            title = paper_metadata.get(paper_id, {}).get("title", "N/A")
            published_date = paper_metadata.get(paper_id, {}).get("published", "N/A")
            
            final_summary = f"# {title}\n\n**Published Date:** {published_date}\n\n---\n\n{summary_content}"

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(final_summary)
            print(f"Summary saved to {summary_path}")
        else:
            print(f"Could not extract text from {pdf_file}. Skipping summarization.")

if __name__ == "__main__":
    arxiv_papers_dir = "arxiv_papers"
    best_prompt_file = "best_prompt.txt"
    asyncio.run(summarize_papers(arxiv_papers_dir, best_prompt_file))
