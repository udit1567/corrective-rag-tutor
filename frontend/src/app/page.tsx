"use client";

import { useState, useEffect } from "react";
import { 
  Search, 
  Database, 
  Globe, 
  Filter, 
  BookOpen, 
  AlertCircle, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  ChevronDown, 
  ChevronUp, 
  Copy, 
  Check,
  Server,
  HelpCircle
} from "lucide-react";

interface StatusData {
  index_created: boolean;
  tavily_configured: boolean;
  documents: Record<string, string>;
  ollama_base_url: string;
}

interface DocDetail {
  page_content: string;
  metadata: {
    source?: string;
    page?: number;
    url?: string;
    title?: string;
  };
}

interface QueryResult {
  question: string;
  verdict: string;
  reason: string;
  docs: DocDetail[];
  good_docs: DocDetail[];
  web_query: string;
  web_docs: DocDetail[];
  strips: string[];
  kept_strips: string[];
  refined_context: string;
  answer: string;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [rebuildingIndex, setRebuildingIndex] = useState(false);
  const [status, setStatus] = useState<StatusData | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  
  // Accordion toggle states
  const [expandedSteps, setExpandedSteps] = useState({
    retrieve: true,
    evaluate: true,
    webSearch: true,
    refine: true,
    generate: true
  });

  const backendUrl = "http://localhost:8000";

  // Fetch status on load
  const fetchStatus = async () => {
    try {
      const res = await fetch(`${backendUrl}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      } else {
        console.error("Failed to load backend status.");
      }
    } catch (err) {
      console.error("Backend server is not running or unreachable:", err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await fetch(`${backendUrl}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Query execution failed.");
      }

      const data = await res.json();
      setResult(data);
      // Auto-expand relevant blocks
      setExpandedSteps({
        retrieve: true,
        evaluate: true,
        webSearch: data.verdict !== "CORRECT",
        refine: true,
        generate: true
      });
    } catch (err: any) {
      setError(err.message || "An error occurred while connecting to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleRebuildIndex = async () => {
    if (!confirm("Are you sure you want to rebuild the vector database? This will process and re-embed the PDF files using Ollama (which may take a moment).")) return;
    
    setRebuildingIndex(true);
    setError(null);
    
    try {
      const res = await fetch(`${backendUrl}/api/rebuild-index`, {
        method: "POST"
      });
      
      if (!res.ok) {
        throw new Error("Failed to rebuild local index.");
      }
      
      alert("Vector store successfully rebuilt and stored locally!");
      await fetchStatus();
    } catch (err: any) {
      setError(err.message || "Failed to rebuild database index.");
    } finally {
      setRebuildingIndex(false);
    }
  };

  const toggleStep = (step: keyof typeof expandedSteps) => {
    setExpandedSteps(prev => ({ ...prev, [step]: !prev[step] }));
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getCleanFilename = (path?: string) => {
    if (!path) return "Unknown Book";
    return path.split("/").pop() || path;
  };

  const runQuickQuery = (queryText: string) => {
    setQuestion(queryText);
    setTimeout(() => {
      const form = document.getElementById("query-form") as HTMLFormElement;
      if (form) form.requestSubmit();
    }, 100);
  };

  return (
    <div className="app-container">
      {/* Header spanning both grid columns */}
      <header className="app-header">
        <div className="brand-section">
          <h1 className="brand-title">
            Corrective <span className="gradient-text">RAG Assistant</span>
          </h1>
          <span className="brand-subtitle">ML Capstone • Visualizing Graph Execution</span>
        </div>
        <div className="stat-item" style={{ gap: "10px", padding: "8px 12px" }}>
          <Server size={16} className="text-muted" />
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Backend: {status ? "Connected" : "Disconnected"}
          </span>
          <div className="pulsing-dot" style={{ backgroundColor: status ? "var(--success)" : "var(--danger)" }}></div>
        </div>
      </header>

      {/* Column 1: Control Panel Sidebar */}
      <aside className="sidebar">
        <section className="panel">
          <h3 className="section-title">Database Status</h3>
          <ul className="stat-list">
            <li className="stat-item">
              <span className="stat-label">
                <Database size={15} /> Local Chroma DB
              </span>
              <span className={`badge ${status?.index_created ? "badge-success" : "badge-danger"}`}>
                {status?.index_created ? "Ready" : "Missing"}
              </span>
            </li>
            <li className="stat-item">
              <span className="stat-label">
                <Globe size={15} /> Tavily Web Search
              </span>
              <span className={`badge ${status?.tavily_configured ? "badge-success" : "badge-neutral"}`}>
                {status?.tavily_configured ? "Active" : "Inactive"}
              </span>
            </li>
          </ul>
          
          <button 
            className="btn btn-outline mt-md" 
            onClick={handleRebuildIndex}
            disabled={rebuildingIndex || loading}
          >
            <RefreshCw size={15} className={rebuildingIndex ? "spin" : ""} style={{ animation: rebuildingIndex ? "spin 1.5s linear infinite" : "" }} />
            {rebuildingIndex ? "Rebuilding Index..." : "Rebuild Database"}
          </button>
        </section>

        <section className="panel">
          <h3 className="section-title">Documents</h3>
          <ul className="stat-list" style={{ gap: "0.75rem" }}>
            {status?.documents ? (
              Object.entries(status.documents).map(([name, state]) => (
                <li key={name} className="stat-item" style={{ padding: "0.5rem 0.75rem" }}>
                  <span className="stat-label" style={{ fontSize: "0.8rem" }}>
                    <BookOpen size={13} /> {name}
                  </span>
                  <span className={`badge ${state === "found" ? "badge-success" : "badge-danger"}`} style={{ fontSize: "0.7rem", padding: "0.15rem 0.4rem" }}>
                    {state}
                  </span>
                </li>
              ))
            ) : (
              <li className="text-muted" style={{ fontSize: "0.85rem" }}>Loading status...</li>
            )}
          </ul>
        </section>

        <section className="panel">
          <h3 className="section-title">Local LLM Stack</h3>
          <ul className="stat-list" style={{ gap: "0.6rem", fontSize: "0.8rem" }}>
            <li className="flex-between">
              <span className="text-muted">Model:</span>
              <span style={{ fontWeight: 600 }}>gemma3:4b</span>
            </li>
            <li className="flex-between">
              <span className="text-muted">Embeddings:</span>
              <span style={{ fontWeight: 600 }}>embeddinggemma</span>
            </li>
            <li className="flex-between">
              <span className="text-muted">Ollama Host:</span>
              <span style={{ fontWeight: 600, fontSize: "0.7rem", color: "var(--primary)" }}>
                {status?.ollama_base_url || "http://localhost:11434"}
              </span>
            </li>
          </ul>
        </section>
      </aside>

      {/* Column 2: Search Console and Graph Visualization */}
      <main className="main-dashboard">
        {/* Query Input Area */}
        <section className="panel">
          <form id="query-form" onSubmit={handleQuery} className="query-box">
            <h3 className="section-title" style={{ marginBottom: "0.5rem" }}>Ask the ML Tutor</h3>
            <p className="text-muted" style={{ fontSize: "0.85rem", marginBottom: "0.75rem" }}>
              Enter a machine learning or artificial intelligence question. The Corrective RAG pipeline will verify the retrieved chunks, fallback to Tavily web search if needed, refine the context sentence-by-sentence, and generate an answer.
            </p>
            <div className="textarea-container">
              <textarea
                className="query-textarea"
                placeholder="Ask about batch normalization vs layer normalization, vanishing gradients, backpropagation..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={loading || rebuildingIndex}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    const form = document.getElementById("query-form") as HTMLFormElement;
                    if (form) form.requestSubmit();
                  }
                }}
              />
            </div>
            
            <div className="flex-between">
              <span className="text-muted" style={{ fontSize: "0.8rem" }}>
                Press Enter to search
              </span>
              <button 
                type="submit" 
                className="btn btn-primary" 
                style={{ width: "auto", minWidth: "150px" }}
                disabled={loading || rebuildingIndex || !question.trim()}
              >
                <Search size={16} />
                Search
              </button>
            </div>
          </form>

          {/* Quick Examples */}
          <div style={{ marginTop: "1rem" }}>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "#fff" }}>Quick Try:</span>
            <div className="queries-grid">
              <button 
                type="button"
                className="quick-query-card" 
                onClick={() => runQuickQuery("What is the difference between batch normalization and layer normalization?")}
                disabled={loading || rebuildingIndex}
              >
                Batch vs Layer Normalization
              </button>
              <button 
                type="button"
                className="quick-query-card" 
                onClick={() => runQuickQuery("Explain how backpropagation works in deep learning")}
                disabled={loading || rebuildingIndex}
              >
                Explain Backpropagation
              </button>
              <button 
                type="button"
                className="quick-query-card" 
                onClick={() => runQuickQuery("What is the primary difference between supervised and unsupervised learning?")}
                disabled={loading || rebuildingIndex}
              >
                Supervised vs Unsupervised
              </button>
            </div>
          </div>
        </section>

        {/* Errors section */}
        {error && (
          <div className="panel" style={{ borderColor: "var(--danger)", display: "flex", gap: "12px", alignItems: "center", background: "rgba(239, 68, 68, 0.05)" }}>
            <AlertCircle color="var(--danger)" size={24} style={{ flexShrink: 0 }} />
            <div>
              <h4 style={{ color: "var(--danger)", fontWeight: 700 }}>Execution Error</h4>
              <p className="text-muted" style={{ fontSize: "0.85rem" }}>{error}</p>
            </div>
          </div>
        )}

        {/* Loading Spinner */}
        {loading && (
          <div className="panel spinner-container">
            <div className="spinner"></div>
            <div className="text-center">
              <h4 style={{ fontWeight: 700 }}>Processing Query through LangGraph...</h4>
              <p className="text-muted" style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>
                Executing steps: Retrieve → Score → Web Fallback (if ambiguous) → Sentence Filtering → Generator
              </p>
            </div>
          </div>
        )}

        {/* C-RAG Trace Output Timeline */}
        {result && !loading && (
          <div className="stepper">
            
            {/* Step 1: RETRIEVAL */}
            <div className={`step-node completed`}>
              <div className="step-icon-container">
                <Database size={18} />
              </div>
              <div className="step-card">
                <div className="step-header" onClick={() => toggleStep("retrieve")}>
                  <div className="step-title-wrapper">
                    <span className="step-title">1. Document Retrieval</span>
                    <span className="badge badge-neutral" style={{ fontSize: "0.7rem" }}>
                      {result.docs.length} Chunks
                    </span>
                  </div>
                  {expandedSteps.retrieve ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
                
                {expandedSteps.retrieve && (
                  <div className="step-body">
                    <p className="text-muted" style={{ fontSize: "0.85rem", marginBottom: "1rem" }}>
                      Queried local Chroma vector database using Gemma embeddings. Retrieved top {result.docs.length} similarity chunks:
                    </p>
                    <div className="docs-grid">
                      {result.docs.map((doc, idx) => (
                        <div key={idx} className="doc-card">
                          <div className="doc-meta">
                            <span className="doc-source">
                              {getCleanFilename(doc.metadata.source)}
                            </span>
                            <span className="doc-page">
                              Page {doc.metadata.page || "?"}
                            </span>
                          </div>
                          <p className="doc-text">"{doc.page_content}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Step 2: EVALUATION */}
            <div className={`step-node ${
              result.verdict === "CORRECT" ? "completed" : 
              result.verdict === "AMBIGUOUS" ? "warning" : "error"
            }`}>
              <div className="step-icon-container">
                {result.verdict === "CORRECT" ? <CheckCircle size={18} /> : 
                 result.verdict === "AMBIGUOUS" ? <AlertCircle size={18} /> : <XCircle size={18} />}
              </div>
              <div className="step-card">
                <div className="step-header" onClick={() => toggleStep("evaluate")}>
                  <div className="step-title-wrapper">
                    <span className="step-title">2. Score-Based Evaluation</span>
                    <span className={`badge ${
                      result.verdict === "CORRECT" ? "badge-success" : 
                      result.verdict === "AMBIGUOUS" ? "badge-warning" : "badge-danger"
                    }`}>
                      {result.verdict}
                    </span>
                  </div>
                  {expandedSteps.evaluate ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
                
                {expandedSteps.evaluate && (
                  <div className="step-body">
                    <div className="stat-item" style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.05)", padding: "1rem", borderRadius: "8px" }}>
                      <div style={{ display: "flex", flexDirection: "column" }}>
                        <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "#fff", marginBottom: "0.25rem" }}>
                          Evaluation Verdict Reason
                        </span>
                        <span className="text-muted" style={{ fontSize: "0.85rem", lineHeight: 1.5 }}>
                          {result.reason}
                        </span>
                      </div>
                    </div>
                    <div style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--text-muted)", display: "flex", gap: "10px" }}>
                      <span>Upper Threshold: <strong>0.7</strong></span>
                      <span>•</span>
                      <span>Lower Threshold: <strong>0.3</strong></span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Step 3: WEB SEARCH SEARCH (If triggered) */}
            {result.verdict !== "CORRECT" && (
              <div className="step-node completed">
                <div className="step-icon-container">
                  <Globe size={18} />
                </div>
                <div className="step-card">
                  <div className="step-header" onClick={() => toggleStep("webSearch")}>
                    <div className="step-title-wrapper">
                      <span className="step-title">3. Query Rewrite & Web Search</span>
                      {result.web_docs.length > 0 ? (
                        <span className="badge badge-success" style={{ fontSize: "0.7rem" }}>
                          {result.web_docs.length} Results
                        </span>
                      ) : (
                        <span className="badge badge-danger" style={{ fontSize: "0.7rem" }}>
                          Skipped/Failed
                        </span>
                      )}
                    </div>
                    {expandedSteps.webSearch ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                  
                  {expandedSteps.webSearch && (
                    <div className="step-body">
                      <div className="stat-item" style={{ background: "rgba(255, 255, 255, 0.025)", padding: "0.75rem 1rem", borderRadius: "8px", marginBottom: "1rem", border: "1px dashed rgba(0, 242, 254, 0.2)" }}>
                        <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                          Rewritten search query: 
                        </span>
                        <strong style={{ marginLeft: "0.5rem", color: "var(--primary)", fontSize: "0.9rem" }}>
                          "{result.web_query}"
                        </strong>
                      </div>
                      
                      {result.web_docs.length > 0 ? (
                        <div className="docs-grid">
                          {result.web_docs.map((doc, idx) => (
                            <div key={idx} className="doc-card" style={{ borderColor: "rgba(0, 242, 254, 0.15)" }}>
                              <div className="doc-meta">
                                <span className="doc-source" style={{ color: "var(--secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "200px" }}>
                                  {doc.metadata.title || "Web Result"}
                                </span>
                                <a 
                                  href={doc.metadata.url} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  className="doc-page"
                                  style={{ color: "var(--primary)", fontSize: "0.7rem", textDecoration: "none" }}
                                >
                                  Link
                                </a>
                              </div>
                              <p className="doc-text" style={{ fontSize: "0.8rem" }}>
                                {doc.page_content.replace(/TITLE:.*?\nURL:.*?\nCONTENT:\n/, "")}
                              </p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ display: "flex", gap: "8px", alignItems: "center", background: "rgba(245, 158, 11, 0.05)", border: "1px solid rgba(245, 158, 11, 0.15)", padding: "0.75rem 1rem", borderRadius: "8px" }}>
                          <AlertCircle size={16} color="var(--warning)" />
                          <span style={{ fontSize: "0.8rem", color: "var(--warning)" }}>
                            {!status?.tavily_configured 
                              ? "Tavily search skipped because TAVILY_API_KEY is not set. Please set the API key in the backend .env configuration." 
                              : "No web results were returned. Using retrieved local document chunks."}
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Step 4: KNOWLEDGE REFINEMENT */}
            <div className="step-node completed">
              <div className="step-icon-container">
                <Filter size={18} />
              </div>
              <div className="step-card">
                <div className="step-header" onClick={() => toggleStep("refine")}>
                  <div className="step-title-wrapper">
                    <span className="step-title">4. Sentence-Level Filtering</span>
                    <span className="badge badge-success" style={{ fontSize: "0.7rem" }}>
                      {result.kept_strips.length} / {result.strips.length} Kept
                    </span>
                  </div>
                  {expandedSteps.refine ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
                
                {expandedSteps.refine && (
                  <div className="step-body">
                    <p className="text-muted" style={{ fontSize: "0.85rem", marginBottom: "1rem" }}>
                      Context document contents were broken down into individual sentences. A binary LLM judge evaluated each sentence independently, filtering out redundant or irrelevant information:
                    </p>
                    
                    {result.strips.length > 0 ? (
                      <div className="sentences-list">
                        {result.strips.map((sentence, idx) => {
                          const isKept = result.kept_strips.includes(sentence);
                          return (
                            <div key={idx} className={`sentence-item ${isKept ? "kept" : "dropped"}`}>
                              <div className="sentence-status-dot"></div>
                              <p style={{ margin: 0 }}>{sentence}</p>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-muted text-center" style={{ fontSize: "0.85rem" }}>
                        No context sentences were extracted for evaluation.
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Step 5: GENERATION & FINAL ANSWER */}
            <div className="step-node completed">
              <div className="step-icon-container">
                <HelpCircle size={18} style={{ color: "var(--primary)" }} />
              </div>
              <div className="step-card final-answer-panel">
                <div className="step-header" onClick={() => toggleStep("generate")}>
                  <div className="step-title-wrapper">
                    <span className="step-title" style={{ color: "var(--primary)" }}>5. Final Tutor Answer</span>
                  </div>
                  {expandedSteps.generate ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
                
                {expandedSteps.generate && (
                  <div className="step-body" style={{ borderTopColor: "rgba(0, 242, 254, 0.15)" }}>
                    <div className="answer-header">
                      <span className="text-muted" style={{ fontSize: "0.8rem" }}>
                        Generated using Ollama (gemma3:4b) based ONLY on refined context:
                      </span>
                      <button 
                        className="btn btn-outline" 
                        onClick={() => handleCopyText(result.answer)}
                        style={{ padding: "0.35rem 0.75rem", fontSize: "0.75rem", width: "auto", borderRadius: "6px" }}
                      >
                        {copied ? (
                          <>
                            <Check size={12} color="var(--success)" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy size={12} />
                            Copy
                          </>
                        )}
                      </button>
                    </div>
                    <div className="answer-body">{result.answer}</div>
                  </div>
                )}
              </div>
            </div>

          </div>
        )}

        {/* Blank state */}
        {!result && !loading && (
          <div className="panel text-center" style={{ padding: "4rem 2rem", borderStyle: "dashed" }}>
            <Database size={48} className="text-muted" style={{ margin: "0 auto 1.5rem", opacity: 0.5 }} />
            <h3 style={{ fontWeight: 700, marginBottom: "0.5rem" }}>No Query Executed Yet</h3>
            <p className="text-muted" style={{ fontSize: "0.9rem", maxWidth: "450px", margin: "0 auto" }}>
              Submit a question to see the Corrective RAG pipeline execute its state-graph workflow and output intermediate reasoning.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
