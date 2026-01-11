from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

# Import RAG agent
from task2 import agent as get_rag_agent

# ---------------------------------
# APP INITIALIZATION
# ---------------------------------
app = FastAPI(
    title="AI Engineer Assignment – Task 3",
    description=(
        "Backend API for querying company policy documents using "
        "Retrieval-Augmented Generation (RAG) with LangChain and OpenAI."
    ),
    version="1.0.0",
)

# Initialize agent once (important)
rag_agent = get_rag_agent

# ---------------------------------
# SCHEMAS
# ---------------------------------
class AskRequest(BaseModel):
    query: str = Field(..., description="User question")
    session_id: Optional[str] = Field(
        default="default_session",
        description="Session ID for conversation memory"
    )

class AskResponse(BaseModel):
    answer: str
    source: List[str]

# ---------------------------------
# HOME / HEALTH ENDPOINT
# ---------------------------------
@app.get("/", tags=["Health"])
def home():
    return {
        "status": "running",
        "message": "AI Engineer Assignment API is up and running.",
        "endpoints": {
            "ask": "POST /ask",
            "docs": "/docs"
        }
    }

# ---------------------------------
# CORE ASK ENDPOINT
# ---------------------------------
@app.post("/ask", response_model=AskResponse, tags=["Query"])
def ask(payload: AskRequest):
    try:
        result = rag_agent.invoke(
            {"messages": [{"role": "user", "content": payload.query}]},
            {"configurable": {"thread_id": payload.session_id}},
        )

        answer = result["messages"][-1].content
        sources = set()

        # Extract sources safely
        for msg in result["messages"]:
            if isinstance(msg, dict) and "artifact" in msg:
                for doc in msg["artifact"]:
                    src = doc.get("metadata", {}).get("source")
                    if src:
                        sources.add(src)

        return {
            "answer": answer,
            "source": list(sources)
        }

    except Exception as e:
        # Controlled error for API consumers
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing query: {str(e)}"
        )
