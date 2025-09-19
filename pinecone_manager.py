import os
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import uuid
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = "company-policies-index"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # or text-embedding-3-large
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
NAMESPACE = "company_policies\Mergestack_Employment Policy Handbook_2025.pdf"

class PineconeVectorStoreManager:
    def __init__(self):
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create index if not exists
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )
            print(f"✅ Created Pinecone index: {PINECONE_INDEX_NAME}")

        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    # -----------------------------
    # Embedding
    # -----------------------------
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            embeddings.append(resp.data[0].embedding)
        return embeddings

    # -----------------------------
    # Upsert
    # -----------------------------
    def upsert_documents(self, docs: List[dict]):
        texts = [d["content"] for d in docs]
        embeddings = self.embed_text(texts)
        to_upsert = [
            (
                str(uuid.uuid4()),
                embeddings[i],
                {"content": d["content"], **d["metadata"]}
            )
            for i, d in enumerate(docs)
        ]
        namespace = docs[0]["metadata"].get("source", "default")
        self.index.upsert(vectors=to_upsert,namespace=namespace)
        print(f"✅ Upserted {len(to_upsert)} vectors into Pinecone")

    # -----------------------------
    # Similarity Search
    # -----------------------------
    def similarity_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for most similar chunks in Pinecone index."""
        query_embedding = self.embed_text([query])[0]

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE  # use the same namespace you used during upsert (e.g. "default" or the source value
        )

        return [
            {
                "content": match["metadata"]["content"],
                "score": match["score"],
                "metadata": {k: v for k, v in match["metadata"].items() if k != "content"}
            }
            for match in results["matches"]
        ]
