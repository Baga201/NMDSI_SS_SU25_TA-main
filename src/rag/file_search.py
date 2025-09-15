"""
PDF file search utilities for RAG agent.
"""
import os
from typing import List

def list_pdfs(data_dir: str) -> List[str]:
    """Return list of PDF file paths under data_dir."""
    pdfs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root, f))
    return pdfs
