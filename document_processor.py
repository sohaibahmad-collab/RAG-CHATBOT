import re
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
import docx


class DocumentProcessor:
    def __init__(self, docs_dir: str = "./company_policies"):
        self.docs_dir = docs_dir
    
    def _clean_text(self, text: str) -> str:
        """Clean up text: collapse multiple newlines (even with spaces), remove underscore lines."""
    # Normalize whitespace-only lines → turn them into plain newlines
        text = re.sub(r"[ \t]*\n", "\n", text)

    # Collapse multiple newlines into one
        text = re.sub(r"\n+", "\n", text)

    # Remove underscore-only lines
        text = re.sub(r"^_+\s*$", "", text, flags=re.MULTILINE)

    # Strip leading/trailing whitespace
        return text.strip()
    # -------------------------
    # Document Loading
    # -------------------------
    def load_single_document(self, file_path: str) -> List[dict]:
        """Load a single document and return as list with one dict"""
        path = Path(file_path)
        text = ""

        if path.suffix.lower() in [".txt", ".md"]:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
        elif path.suffix.lower() in [".docx", ".doc"]:
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        text = self._clean_text(text)


        return [{"content": text, "source": str(path)}]

    # -------------------------
    # Hybrid Splitting
    # -------------------------
    def split_by_headings(self, text: str) -> List[str]:
        """Split text by common policy headings (1., 2., Section A, Policy:, etc.)"""
        sections = re.split(r"\n(?=\d+\.|\w+\sPolicy|\w+\sSection|\bChapter\b|\bArticle\b)", text)
        return [s.strip() for s in sections if s.strip()]

    def recursive_split(self, text: str, chunk_size=1000, chunk_overlap=150) -> List[str]:
        """Recursively split text into chunks without breaking sentences."""

        # If already small enough, keep as is
        if len(text) <= chunk_size:
            return [text]

        # Step 1: Sentence split (never break inside sentences)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current_chunk, current_len = [], [], 0

        for sentence in sentences:
            sent_len = len(sentence)

            # If adding this sentence exceeds the chunk size, flush the current chunk
            if current_len + sent_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Apply sentence-level overlap
                overlap_text = current_chunk[-(chunk_overlap // 20):]  # keep last few sentences
                current_chunk = overlap_text if overlap_text else []
                current_len = sum(len(s) + 1 for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def hybrid_split(self, text: str, chunk_size=1000, chunk_overlap=150) -> List[str]:
        """Combine heading-based + recursive splitting"""
        sections = self.split_by_headings(text)
        chunks = []
        for section in sections:
            chunks.extend(self.recursive_split(section, chunk_size, chunk_overlap))
        return chunks

    # -------------------------
    # Prepare Final Docs
    # -------------------------
    def prepare_documents(self, docs: List[dict], chunk_size=1000, chunk_overlap=150) -> List[dict]:
        """Prepare docs using hybrid chunking"""
        chunked_docs = []
        for doc in docs:
            chunks = self.hybrid_split(doc["content"], chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "content": chunk,
                    "metadata": {"source": doc["source"], "chunk": i}
                })
        return chunked_docs
   