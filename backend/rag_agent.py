import os
import json
import re
import math
from typing import List, TypedDict, Dict, Any
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Global configuration
CHROMA_DB_PATH = "chroma_db"
UPPER_TH = 0.7
LOWER_TH = 0.3


print("Initializing Ollama models...")
embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
llm = ChatOllama(model="gemma3:4b", temperature=0)

# Document retrieval models & states
class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str

# Structured output Pydantic schemas
class DocEvalScore(BaseModel):
    score: float = Field(description="Relevance score in [0.0, 1.0]")
    reason: str = Field(description="Brief explanation of the score")

class KeepOrDrop(BaseModel):
    keep: bool = Field(description="Whether to keep the sentence (true) or drop it (false)")

class WebQuery(BaseModel):
    query: str = Field(description="Rewritten web search query")

# Helper function to parse JSON safely
def parse_json_safely(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try finding markdown JSON block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Try finding any outer braces
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"Could not parse JSON from text: {text}")


# LLM Chains and Fallback logic ------------


# 1. Document Evaluator
doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON with keys 'score' (float) and 'reason' (string). Output ONLY raw JSON.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

def evaluate_document(question: str, chunk: str) -> DocEvalScore:
    try:
        chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
        return chain.invoke({"question": question, "chunk": chunk})
    except Exception as e:
        print(f"Fallback doc evaluation: {e}")
        # Manual fallback parsing
        formatted = doc_eval_prompt.format_messages(question=question, chunk=chunk)
        res_raw = llm.invoke(formatted)
        try:
            parsed = parse_json_safely(res_raw.content)
            return DocEvalScore(score=float(parsed.get("score", 0.0)), reason=str(parsed.get("reason", "")))
        except Exception as ex:
            print(f"Doc evaluation fallback parsing failed: {ex}")
            return DocEvalScore(score=0.5, reason="Parsed failed, default score assigned.")

# 2. Keep or Drop Filter
filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON with a key 'keep' (boolean). Output ONLY raw JSON.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

def should_keep_sentence(question: str, sentence: str) -> KeepOrDrop:
    try:
        chain = filter_prompt | llm.with_structured_output(KeepOrDrop)
        return chain.invoke({"question": question, "sentence": sentence})
    except Exception as e:
        print(f"Fallback sentence filter: {e}")
        formatted = filter_prompt.format_messages(question=question, sentence=sentence)
        res_raw = llm.invoke(formatted)
        try:
            parsed = parse_json_safely(res_raw.content)
            val = parsed.get("keep", False)
            if isinstance(val, str):
                val = val.lower() == "true"
            return KeepOrDrop(keep=bool(val))
        except Exception as ex:
            print(f"Sentence filter fallback parsing failed: {ex}")
            return KeepOrDrop(keep=True)

# 3. Query Rewriter
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: 'query' (string). Output ONLY raw JSON.",
        ),
        ("human", "Question: {question}"),
    ]
)

def rewrite_question_for_web(question: str) -> WebQuery:
    try:
        chain = rewrite_prompt | llm.with_structured_output(WebQuery)
        return chain.invoke({"question": question})
    except Exception as e:
        print(f"Fallback query rewrite: {e}")
        formatted = rewrite_prompt.format_messages(question=question)
        res_raw = llm.invoke(formatted)
        try:
            parsed = parse_json_safely(res_raw.content)
            return WebQuery(query=str(parsed.get("query", question)))
        except Exception as ex:
            print(f"Query rewrite fallback parsing failed: {ex}")
            return WebQuery(query=question)


# Vector Store Management -----------------------

vector_store = None
retriever = None

def init_vector_store(force_rebuild: bool = False) -> bool:
    global vector_store, retriever
    
    # Check if the database folder AND sqlite database file exists (to avoid reading empty folders)
    sqlite_db_file = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    db_exists = os.path.exists(CHROMA_DB_PATH) and os.path.exists(sqlite_db_file)

    if not force_rebuild and db_exists:
        try:
            print("Loading local Chroma vector store...")
            vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )
            count = vector_store._collection.count()
            print(f"Chroma vector store loaded. Documents found: {count}")
            if count > 0:
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                return True
            else:
                print("Chroma database is empty. Triggering automatic index rebuild...")
        except Exception as e:
            print(f"Error loading Chroma database: {e}. Rebuilding...")

    print("Rebuilding Chroma index from PDF documents...")
    
    # PDF paths now look inside the Docs directory
    pdf_names = ["book1.pdf", "book2.pdf", "book3.pdf"]
    pdf_paths = []
    
    for name in pdf_names:
        p_parent = os.path.join("..", "Docs", name)
        p_curr = os.path.join(".", "Docs", name)
        p_abs = os.path.join("/Volumes/SSD/C-RAG/Docs", name)
        
        if os.path.exists(p_parent):
            pdf_paths.append(p_parent)
        elif os.path.exists(p_curr):
            pdf_paths.append(p_curr)
        elif os.path.exists(p_abs):
            pdf_paths.append(p_abs)
            
    if len(pdf_paths) < 3:
        if not pdf_paths:
            raise FileNotFoundError("Could not find book1.pdf, book2.pdf, or book3.pdf in parent/Docs, current/Docs, or absolute /Volumes/SSD/C-RAG/Docs directories.")
        print(f"Warning: Found only {len(pdf_paths)} documents: {pdf_paths}. Proceeding anyway.")

    docs = []
    for path in pdf_paths:
        try:
            print(f"Reading document: {path}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    if not docs:
        raise ValueError("Could not extract any content from PDF files.")

    print(f"Splitting {len(docs)} pages into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # If force rebuild and directory exists, delete it first to avoid duplicate documents
    if force_rebuild and os.path.exists(CHROMA_DB_PATH):
        import shutil
        try:
            print("Removing existing Chroma directory...")
            # If it's a symlink, remove the contents of the target directory to clear APFS storage
            real_path = os.path.realpath(CHROMA_DB_PATH)
            if os.path.isdir(real_path):
                for filename in os.listdir(real_path):
                    file_path = os.path.join(real_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                shutil.rmtree(CHROMA_DB_PATH)
        except Exception as e:
            print(f"Error removing Chroma database contents: {e}")

    # To avoid overloading local Ollama embeddings endpoint, initialize with first chunk and batch remaining
    print(f"Initializing ChromaDB on disk with first chunk out of {len(chunks)}...")
    vector_store = Chroma.from_documents(
        documents=[chunks[0]],
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    # Process remaining chunks in batches of 100
    batch_size = 100
    remaining_chunks = chunks[1:]
    total_batches = math.ceil(len(remaining_chunks) / batch_size)
    print(f"Generating embeddings using Ollama {embeddings.model} in {total_batches} batches of size {batch_size}...")
    
    for i in range(0, len(remaining_chunks), batch_size):
        batch = remaining_chunks[i : i + batch_size]
        print(f"Adding batch {i // batch_size + 1} of {total_batches} (size: {len(batch)})...")
        vector_store.add_documents(batch)

    # Persist the database
    try:
        vector_store.persist()
    except AttributeError:
        pass
        
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("Chroma vector store indexed and saved locally.")
    return True


# Graph Node Definitions -----------------------

def retrieve_node(state: State) -> Dict[str, Any]:
    print("--> Entering Node: retrieve")
    q = state["question"]
    global retriever
    if retriever is None:
        init_vector_store()
    
    docs = retriever.invoke(q) if retriever else []
    print(f"Retrieved {len(docs)} documents.")
    return {"docs": docs}

def eval_each_doc_node(state: State) -> Dict[str, Any]:
    print("--> Entering Node: eval_each_doc")
    q = state["question"]
    docs = state.get("docs", [])
    scores: List[float] = []
    good: List[Document] = []

    for d in docs:
        out = evaluate_document(q, d.page_content)
        scores.append(out.score)
        print(f"Evaluated chunk (Score: {out.score}): {out.reason}")
        if out.score > LOWER_TH:
            good.append(d)

    # Verdict routing logic
    if any(s > UPPER_TH for s in scores):
        verdict = "CORRECT"
        reason = f"At least one retrieved chunk scored > {UPPER_TH}."
    elif len(scores) > 0 and all(s < LOWER_TH for s in scores):
        verdict = "INCORRECT"
        reason = f"All retrieved chunks scored < {LOWER_TH}."
    else:
        verdict = "AMBIGUOUS"
        reason = f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}."

    print(f"Verdict: {verdict} ({reason})")
    return {
        "good_docs": good,
        "verdict": verdict,
        "reason": reason,
    }

def rewrite_query_node(state: State) -> Dict[str, Any]:
    print("--> Entering Node: rewrite_query")
    q = state["question"]
    out = rewrite_question_for_web(q)
    print(f"Rewrote query for web search: '{out.query}'")
    return {"web_query": out.query}

def web_search_node(state: State) -> Dict[str, Any]:
    print("--> Entering Node: web_search")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("Warning: TAVILY_API_KEY environment variable is not set. Skipping web search fallback.")
        return {"web_docs": []}

    q = state.get("web_query") or state["question"]
    try:
        tavily = TavilySearchResults(max_results=5)
        results = tavily.invoke({"query": q})
        web_docs: List[Document] = []
        for r in results or []:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        print(f"Web search returned {len(web_docs)} documents.")
        return {"web_docs": web_docs}
    except Exception as e:
        print(f"Error during Tavily web search: {e}")
        return {"web_docs": []}

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def refine(state: State) -> Dict[str, Any]:
    print("--> Entering Node: refine")
    q = state["question"]
    verdict = state.get("verdict", "AMBIGUOUS")

    if verdict == "CORRECT":
        docs_to_use = state.get("good_docs", [])
    elif verdict == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else:  # AMBIGUOUS
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    if not context:
        print("Context is empty. Skipping filtering.")
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }

    strips = decompose_to_sentences(context)
    print(f"Decomposed context into {len(strips)} sentences. Filtering...")

    kept: List[str] = []
    for s in strips:
        out = should_keep_sentence(q, s)
        if out.keep:
            kept.append(s)

    refined_context = "\n".join(kept).strip()
    print(f"Filtering complete. Retained {len(kept)}/{len(strips)} sentences.")
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def generate(state: State) -> Dict[str, Any]:
    print("--> Entering Node: generate")
    q = state["question"]
    context = state.get("refined_context", "").strip()
    
    res = (answer_prompt | llm).invoke({"question": q, "context": context})
    print("Answer generated.")
    return {"answer": res.content}

# Graph Assembly -----------------------

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()

def run_rag_agent(question: str) -> Dict[str, Any]:
    initial_state = {
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }
    return app.invoke(initial_state)
