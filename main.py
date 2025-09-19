import chainlit as cl
from openai import OpenAI
from pinecone_manager import PineconeVectorStoreManager  # your retriever class

# Initialize retriever + OpenAI client
VECTOR_STORE_MANAGER = PineconeVectorStoreManager()
openai_client = OpenAI()

LLM_MODEL_NAME = "gpt-4o-mini"  # or gpt-4.1 / gpt-3.5-turbo

# Store chat history (role + content)
chat_history = []


# -------------------------
# Chainlit Events
# -------------------------
@cl.on_chat_start
async def start_chat():
    chat_history.clear()  # reset history on new chat
    await cl.Message(content="🤖 Hi! Ask me anything about company HR policies.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content

    # Step 1: Retrieve docs from Pinecone
    retrieved_docs = VECTOR_STORE_MANAGER.similarity_search(query, top_k=40)

    # Step 2: Build context
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    # Add user query to history
    chat_history.append({"role": "user", "content": query})

    # Keep only last 5 exchanges
    truncated_history = chat_history[-5:]

    # Step 3: Prepare OpenAI messages with HR-focused system prompt
    system_prompt = f"""
    You are a helpful HR assistant. 
    Answer employee questions strictly based on the provided company HR policy documents. 

    Guidelines:
    - If the answer is found in the context, explain it clearly and concisely. 
    - If the answer is not found in the context, politely respond that you don’t know 
      and suggest contacting HR for clarification. 
    - Do NOT make up information or answer outside the provided HR policies. 
    - Keep responses professional, supportive, and easy to understand. 

    Context:
    {context}
    """

    messages = [
        {"role": "system", "content": system_prompt}
    ] + truncated_history

    # Step 4: OpenAI completion
    response = openai_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
    )
    answer = response.choices[0].message.content

    # Save assistant reply into history
    chat_history.append({"role": "assistant", "content": answer})

    # Step 5: Send answer back to UI
    await cl.Message(content=answer).send()
