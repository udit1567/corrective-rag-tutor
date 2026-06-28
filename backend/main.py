from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List, Dict, Any

# Import C-RAG agent
from rag_agent import run_rag_agent, init_vector_store, CHROMA_DB_PATH

app = FastAPI(title="Corrective RAG (C-RAG) API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits access from local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    verdict: str
    reason: str
    docs: List[Dict[str, Any]]
    good_docs: List[Dict[str, Any]]
    web_query: str
    web_docs: List[Dict[str, Any]]
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    answer: str

def serialize_document(doc) -> Dict[str, Any]:
    return {
        "page_content": getattr(doc, "page_content", ""),
        "metadata": getattr(doc, "metadata", {})
    }

@app.get("/api/status")
def get_status():
    sqlite_db_file = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    index_exists = os.path.exists(CHROMA_DB_PATH) and os.path.exists(sqlite_db_file)
    tavily_set = bool(os.getenv("TAVILY_API_KEY"))
    
    docs_status = {}
    for name in ["book1.pdf", "book2.pdf", "book3.pdf"]:
        found = False
        for p in [
            os.path.join("..", "Docs", name),
            os.path.join(".", "Docs", name),
            os.path.join("/Volumes/SSD/C-RAG/Docs", name)
        ]:
            if os.path.exists(p):
                found = True
                break
        docs_status[name] = "found" if found else "missing"

    return {
        "index_created": index_exists,
        "tavily_configured": tavily_set,
        "documents": docs_status
    }

@app.post("/api/rebuild-index")
def rebuild_index():
    try:
        init_vector_store(force_rebuild=True)
        return {"status": "success", "message": "Local Chroma vector store has been rebuilt successfully."}
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        print(f"\n--- [QUERY] {req.question} ---")
        res = run_rag_agent(req.question)
        
        # Serialize LangChain Document objects
        docs = [serialize_document(d) for d in res.get("docs", [])]
        good_docs = [serialize_document(d) for d in res.get("good_docs", [])]
        web_docs = [serialize_document(d) for d in res.get("web_docs", [])]
        
        return {
            "question": res.get("question", req.question),
            "verdict": res.get("verdict", "AMBIGUOUS"),
            "reason": res.get("reason", "No evaluation completed."),
            "docs": docs,
            "good_docs": good_docs,
            "web_query": res.get("web_query", ""),
            "web_docs": web_docs,
            "strips": res.get("strips", []),
            "kept_strips": res.get("kept_strips", []),
            "refined_context": res.get("refined_context", ""),
            "answer": res.get("answer", "I don't know.")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error executing C-RAG query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
