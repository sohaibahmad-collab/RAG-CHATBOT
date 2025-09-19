import sys
from document_processor import DocumentProcessor
from pinecone_manager import PineconeVectorStoreManager



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python upload_and_index.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Initialize helpers
    processor = DocumentProcessor()
    pinecone_manager = PineconeVectorStoreManager()

    print(f"📂 Loading document: {file_path}")
    docs = processor.load_single_document(file_path)

    print("✂️ Splitting into chunks...")
    chunked_docs = processor.prepare_documents(docs)

    print("📤 Upserting into Pinecone...")
    pinecone_manager.upsert_documents(chunked_docs)

    print("✅ Done!")
