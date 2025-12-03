import uvicorn
import requests
import json
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware # NEW: Import CORS middleware

# --- NOTE: Replace with your actual Gemini API Key ---
GEMINI_API_KEY = "AIzaSyA_lqdpx7x6XL6CeiFVqX9e17zgdn1EsMk"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# --- Mock Vector Store / Document Storage (In-memory for PoC) ---
# Key: documentId (str), Value: List of document chunks (str)
MOCK_VECTOR_STORE: Dict[str, List[str]] = {}

app = FastAPI(title="RAG PoC Backend")

# --- CORS Middleware Configuration (MANDATORY for Cross-Origin Calls) ---
# Allows the React component (on the Drupal domain) to communicate with this API.
# In a real environment, replace "*" with your specific Drupal domain for security.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Request/Response ---

class IngestRequest(BaseModel):
    pdfUrl: str

class QueryRequest(BaseModel):
    documentId: str
    question: str
    history: List[Dict[str, str]] = [] # Optional chat history

class APIResponse(BaseModel):
    documentId: str = None
    answer: str = None
    status: str

# --- CORE RAG FUNCTIONS (Mocked/Conceptual) ---

def fetch_and_extract_pdf_text(url: str) -> str:
    """
    CONCEPTUAL: Fetches the PDF from the URL and extracts text.
    """
    print(f"-> Attempting to fetch and extract PDF from: {url}")
    
    # --- MOCK IMPLEMENTATION START ---
    doc_id = str(uuid.uuid4())
    
    # Simple document text for demonstration
    mock_document_text = f"""
    [Document ID: {doc_id}]
    The company's proposal for Q3 focuses on three core initiatives: Project Titan, 
    Project Zenith, and Project Echo. Project Titan is allocated $500,000 and is 
    expected to complete by the end of September. Project Zenith is a longer-term 
    R&D effort focused on sustainable energy, with a projected timeline of 18 months 
    and a budget of $2.5 million. The key performance indicator (KPI) for Zenith is 
    a 20% efficiency increase in battery prototypes. Project Echo, valued at $300,000, 
    is a market research study scheduled for July and August. The lead contact for 
    Project Titan is Jane Doe. All projects must adhere to the new security protocols 
    outlined in Section 4. 
    """
    
    return mock_document_text.strip()
    # --- MOCK IMPLEMENTATION END ---


def chunk_and_embed_text(text: str) -> List[str]:
    """
    CONCEPTUAL: Chunks the text and creates embeddings (vectors).
    """
    # --- MOCK IMPLEMENTATION START ---
    chunks = [c.strip() for c in text.split('.') if c.strip()]
    return chunks
    # --- MOCK IMPLEMENTATION END ---


def retrieve_relevant_context(doc_id: str, question: str) -> str:
    """
    CONCEPTUAL: Finds the most relevant document chunks based on the question.
    """
    # --- MOCK IMPLEMENTATION START ---
    if doc_id not in MOCK_VECTOR_STORE:
        return ""
    
    document_chunks = MOCK_VECTOR_STORE[doc_id]
    
    # Trivial keyword matching for PoC
    relevant_chunks = [
        chunk for chunk in document_chunks 
        if any(word.lower() in chunk.lower() for word in question.split())
    ]

    # If no keywords match, just return the first two chunks as mock context
    if not relevant_chunks:
        relevant_chunks = document_chunks[:2]
        
    return "\n---\n".join(relevant_chunks)
    # --- MOCK IMPLEMENTATION END ---


def generate_rag_answer(context: str, question: str, history: List[Dict[str, str]]) -> str:
    """
    ACTUAL API CALL: Constructs the augmented prompt and calls the Gemini API.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return "ERROR: Gemini API key is not configured in the backend server."

    system_prompt = (
        "You are a Proposal Analyst AI. Your task is to answer the user's question "
        "using ONLY the provided CONTEXT. Do not use external knowledge. "
        "If the answer is not found in the context, state 'I cannot find that information in the document.' "
        "Be concise and professional."
    )
    
    rag_prompt = f"CONTEXT:\n---\n{context}\n---\nQUESTION: {question}"

    # Build the message history
    api_messages = []
    
    # Add system instruction
    api_messages.append({"role": "system", "parts": [{"text": system_prompt}]})

    # Add RAG prompt (User role with all context/question)
    api_messages.append({"role": "user", "parts": [{"text": rag_prompt}]})
    
    payload = {
        "contents": api_messages,
        "systemInstruction": system_prompt, # Set as system instruction for Gemini API
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/models/gemini-2.5-flash-preview-09-2025:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the response text
        if data.get('candidates') and data['candidates'][0].get('content'):
             return data['candidates'][0]['content']['parts'][0]['text']
        else:
            return "AI failed to generate a coherent response."

    except requests.exceptions.RequestException as e:
        print(f"Gemini API Request Error: {e}")
        return f"Gemini API Error: Could not connect or received an error."


# --- API Endpoints ---

@app.post("/api/ingest", response_model=APIResponse)
async def ingest_document(request: IngestRequest):
    """
    Endpoint 1: Ingests a PDF document URL, processes it, and stores the chunks.
    """
    try:
        # 1. Fetch & Extract Text
        document_text = fetch_and_extract_pdf_text(request.pdfUrl)
        
        # 2. Chunk & Embed (Mocked)
        chunks = chunk_and_embed_text(document_text)
        
        # 3. Generate unique ID and store (Mocked)
        # Note: We use the first 8 characters of a UUID for the mock ID
        doc_id = str(uuid.uuid4())[:8] 
        MOCK_VECTOR_STORE[doc_id] = chunks
        
        print(f"-> Successfully indexed document {doc_id} with {len(chunks)} chunks.")
        
        return APIResponse(
            documentId=doc_id,
            status="success"
        )
    except Exception as e:
        print(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")

@app.post("/api/query", response_model=APIResponse)
async def query_document(request: QueryRequest):
    """
    Endpoint 2: Performs RAG search and returns a grounded answer.
    """
    doc_id = request.documentId
    
    if doc_id not in MOCK_VECTOR_STORE:
        raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found.")

    try:
        # 1. Retrieval
        context = retrieve_relevant_context(doc_id, request.question)
        
        if not context:
            answer = "I cannot find any relevant sections for that query in the document."
        else:
            # 2. Generation (Actual Gemini API Call)
            answer = generate_rag_answer(context, request.question, request.history)
        
        return APIResponse(
            documentId=doc_id,
            answer=answer,
            status="success"
        )
    except Exception as e:
        print(f"Query Error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

# To run the server: uvicorn rag_api_server:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
