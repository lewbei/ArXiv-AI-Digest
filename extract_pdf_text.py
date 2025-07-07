#!/usr/bin/env python3
"""
Simple PDF text extraction utility
"""

import sys

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
    except ImportError:
        print("PyPDF2 not installed. Trying alternative method...")
        
        try:
            import pdfplumber
            
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip()
            
        except ImportError:
            print("No PDF libraries available.")
            print("Please install: pip install PyPDF2 or pip install pdfplumber")
            print("Or manually copy the text from the PDF and save as .txt file")
            return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf_text.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print(f"Extracting text from: {pdf_path}")
    text = extract_pdf_text(pdf_path)
    
    if text:
        # Save to text file
        txt_path = pdf_path.replace('.pdf', '_extracted.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Text extracted and saved to: {txt_path}")
        print(f"Text length: {len(text)} characters")
        print(f"Word count: {len(text.split())} words")
        
        # Show first few lines
        lines = text.split('\n')[:10]
        print("\nFirst few lines:")
        for line in lines:
            if line.strip():
                print(f"  {line.strip()[:80]}...")
    
    else:
        print("Failed to extract text from PDF")


if __name__ == "__main__":
    main()