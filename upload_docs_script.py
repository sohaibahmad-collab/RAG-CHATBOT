import sys
from document_processor import DocumentProcessor


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python upload_and_index.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Initialize helpers
    processor = DocumentProcessor()
    
    print("📤 Upserting into Pinecone...")
    processor.process_and_upsert(file_path)

    print("✅ Done!")
