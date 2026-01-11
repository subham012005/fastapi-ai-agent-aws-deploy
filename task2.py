import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

model= ChatOpenAI(model="gpt-4", temperature=0)

directory_path = (
    "D:\Vibe_CODEING\Assignment_first500days\document_for_rag"
)

loader = PyPDFDirectoryLoader("document_for_rag/")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # chunk size (characters)
    chunk_overlap=30,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

chunk = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(chunk,embeddings)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You are the First500Days Corporate Assistant, specialized in retrieving information from company policy PDFs.\n\n"
    "OPERATIONAL PROTOCOL:\n"
    "1. **Policy Queries**: Use the `retrieve_context` tool for any questions regarding company rules, benefits, or procedures.\n"
    "2. **Missing Information**: If the tool returns no relevant context, inform the user: 'No relevant information found. Please contact HR for further assistance.'\n"
    "3. **General Interaction**: Respond to greetings and non-policy questions in a highly professional, formal, and helpful tone.\n"
    "4. **Persona**: Maintain a consistent corporate identity. Do not speculate beyond the provided context for policy matters."
)
agent = create_agent(model, tools, system_prompt=prompt, checkpointer=InMemorySaver())

def main(query: str):
    # for event in agent.stream(
    #     {"messages": [{"role": "user", "content": query}]},
    #     {"configurable": {"thread_id": "policy_retrieval_thread"}},
    #     stream_mode="values",
    # ):
    #     event["messages"][-1].pretty_print()
    response = agent.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"default"}}
    )
    print(response)
    
def run():
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        main(query)
        
        
if __name__ == "__main__":
    run()