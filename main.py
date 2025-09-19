import chainlit as cl
from openai import OpenAI
from pinecone_manager import PineconeVectorStoreManager  # your retriever class

# Initialize retriever + OpenAI client
VECTOR_STORE_MANAGER = PineconeVectorStoreManager()
openai_client = OpenAI()

LLM_MODEL_NAME = "gpt-4o-mini"  # or gpt-4.1 / gpt-3.5-turbo
# -------------------------
# Chainlit Events
# -------------------------
@cl.on_chat_start
async def start_chat():
    await cl.Message(content="🤖 Hi! Ask me anything about company policies.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content

    # Step 1: Retrieve docs from Pinecone
    retrieved_docs = VECTOR_STORE_MANAGER.similarity_search(query, top_k=30)

    # Step 2: Inline OpenAI call with retrieved context
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    print(context)
    print("hello")
    prompt = f"""
You are a helpful assistant. Use the following context to answer the user’s question. 
If the answer is not in the context, say you don’t know.

Context:
{context}

Question: {query}
Answer:
"""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL_NAME,  # or gpt-4.1 / gpt-3.5-turbo
        messages=[{"role": "system", "content": "You are a helpful assistant that only answers based on company policies. If the answer is not found in the context, politely say you don’t know."},{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    # Step 3: Send answer back to UI
    await cl.Message(content=answer).send()
