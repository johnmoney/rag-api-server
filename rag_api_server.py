import uvicorn
import requests
import json
import uuid
import os # NEW: Import os for environment variable access
from io import BytesIO # NEW: Import BytesIO for in-memory file handling
from pypdf import PdfReader # NEW: Library for PDF reading/extraction. Requires 'pypdf' in requirements.txt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware # Need to explicitly import for Render deployment

# --- API Configuration ---
# UPDATED: Read GEMINI_API_KEY from the environment variable (Render config)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "MISSING_KEY") 
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# --- Mock Vector Store / Document Storage (In-memory for PoC) ---
MOCK_VECTOR_STORE: Dict[str, List[str]] = {}

app = FastAPI(title="RAG PoC Backend")

# --- CORS Configuration ---
origins = ["*"] # Allow all origins for demo/Canvas
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
    # Note: Frontend must send sanitized history: List[{"role": str, "content": str}]
    history: List[Dict[str, str]] = [] 

class APIResponse(BaseModel):
    documentId: str = None
    answer: str = None
    status: str

# --- CORE RAG FUNCTIONS ---

def fetch_and_extract_pdf_text(url: str) -> str:
    """
    REAL IMPLEMENTATION: Fetches the PDF from the URL and extracts text.
    
    Raises:
        HTTPException: If the PDF download or parsing fails.
    """
    print(f"-> Attempting to fetch and extract PDF from: {url}")
    
    try:
        # 1. Download the PDF content
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() # Check for bad status codes (4xx, 5xx)
        
        # 2. Use BytesIO to treat the downloaded binary content as a file
        file_stream = BytesIO(response.content)
        
        # 3. Extract text using pypdf
        reader = PdfReader(file_stream)
        text = ""
        
        # Extract text page by page
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise ValueError("PDF extraction resulted in empty content.")

        print(f"-> Successfully extracted {len(text)} characters from the PDF.")
        return text.strip()

    except requests.exceptions.RequestException as e:
        # Catch download errors (timeout, connection, 404, etc.)
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to download PDF from URL: {e}"
        )
    except Exception as e:
        # Catch PDF parsing errors
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from PDF: {e}"
        )


def chunk_and_embed_text(text: str) -> List[str]:
    """
    MOCK: Chunks the text.
    (In a real application, a library like LangChain TextSplitter would be used here.)
    """
    # --- MOCK IMPLEMENTATION START (now using real input text) ---
    # Since we don't know the exact length, we'll split by double newline as a basic chunking technique
    chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
    
    # Fallback if double newline split fails (e.g., highly compressed text)
    if len(chunks) < 2 and len(text) > 1000:
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
    return chunks
    # --- MOCK IMPLEMENTATION END ---


def retrieve_relevant_context(doc_id: str, question: str) -> str:
    """
    MOCK: Finds the most relevant document chunks based on the question.
    (In a real application, vector search would occur here.)
    """
    # --- MOCK IMPLEMENTATION START ---
    if doc_id not in MOCK_VECTOR_STORE:
        return ""
    
    document_chunks = MOCK_VECTOR_STORE[doc_id]
    
    # Trivial keyword matching for PoC
    question_words = set(word.lower() for word in question.split() if len(word) > 3)
    
    relevant_chunks = []
    
    # Select chunks that contain any of the question words
    for chunk in document_chunks:
        if any(q_word in chunk.lower() for q_word in question_words):
            relevant_chunks.append(chunk)

    # If no keywords match, just return the first chunk as mock context
    if not relevant_chunks:
        relevant_chunks = document_chunks[:1]
        
    # Limit to a maximum of 3 chunks for context window management
    return "\n---\n".join(relevant_chunks[:3])
    # --- MOCK IMPLEMENTATION END ---


def generate_rag_answer(context: str, question: str, history: List[Dict[str, str]]) -> str:
    """
    ACTUAL API CALL: Constructs the augmented prompt and calls the Gemini API.
    """
    if GEMINI_API_KEY == "MISSING_KEY": 
        return "ERROR: Gemini API key failed to load from environment variables."

    system_prompt = (
        "You are a Proposal Analyst AI. Your primary goal is to provide accurate, concise, "
        "and direct answers."
    )
    
    # --- FIX: Strengthened RAG Prompt Template to enforce extraction ---
    rag_prompt = f"""
DOCUMENT CONTEXT:
---
{context}
---

USER QUESTION: {question}

INSTRUCTION: Based ONLY on the DOCUMENT CONTEXT above, provide a concise answer to the USER QUESTION. 
Do not elaborate or use external knowledge. If the answer is not present in the context, state: 
"I cannot find that information in the document."

ANSWER:
"""

    # Build the message history for the 'contents' array
    api_messages = []

    # 1. Add previous conversation history (Sanitized history from frontend)
    for msg in history:
        # FastAPI frontend history schema: {"role": str, "content": str}
        role = msg['role'].lower()
        
        # --- FIX: Map 'ai' role to 'model' and filter out 'system' roles ---
        if role == 'ai':
            role = 'model'
        elif role == 'system':
            continue # Skip system/internal messages from history
        
        # Gemini API schema: {"role": str, "parts": [{"text": str}]}
        api_messages.append({"role": role, "parts": [{"text": msg['content']}]})
    
    # 2. Add the current RAG-augmented prompt as the final user message
    api_messages.append({"role": "user", "parts": [{"text": rag_prompt}]})
    
    payload = {
        "contents": api_messages,
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]}, 
    }
    
    # Base URL without the key, for logging
    log_url = f"{API_BASE_URL}/models/gemini-2.5-flash-preview-09-2025:generateContent"

    try:
        response = requests.post(
            log_url, # Use URL without key
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            data=json.dumps(payload),
            timeout=15 
        )
        
        # --- NEW ERROR HANDLING START ---
        if response.status_code >= 400:
            # Log the status code and response body (which usually contains the error reason)
            error_detail = response.text 
            print(f"Gemini API Error (Status: {response.status_code}): {error_detail}")
            # Raise generic exception without the URL
            response.raise_for_status() 
        # --- NEW ERROR HANDLING END ---
        
        data = response.json()
        
        # Extract the response text
        if data.get('candidates') and data['candidates'][0].get('content'):
             return data['candidates'][0]['content']['parts'][0]['text']
        else:
            # If API returns success but no content, log and return error
            print(f"Gemini API returned no candidates: {data}")
            return "AI failed to generate a coherent response."

    except requests.exceptions.RequestException as e:
        # This catches connection errors, DNS failure, and status code errors from raise_for_status()
        print(f"Gemini API Request Error: {e}")
        # Return the status code and text if available, avoiding the full URL object 'e'
        if hasattr(e, 'response') and e.response is not None:
             # This message is sent back to the client
             error_message = f"HTTP {e.response.status_code} - See server logs for detail."
        else:
             error_message = f"Network failure: Check server connection/DNS."
             
        return f"Gemini API Request Error: {error_message}"


# --- API Endpoints ---

@app.post("/api/ingest", response_model=APIResponse)
async def ingest_document(request: IngestRequest):
    """
    Endpoint 1: Ingests a PDF document URL, processes it, and stores the chunks.
    """
    try:
        # 1. Fetch & Extract REAL Text
        document_text = fetch_and_extract_pdf_text(request.pdfUrl)
        
        # 2. Chunk & Embed (Mocked)
        chunks = chunk_and_embed_text(document_text)
        
        # 3. Generate unique ID and store (Mocked)
        doc_id = str(uuid.uuid4())[:8] 
        MOCK_VECTOR_STORE[doc_id] = chunks
        
        print(f"-> Successfully indexed document {doc_id} with {len(chunks)} chunks.")
        
        return APIResponse(
            documentId=doc_id,
            status="success"
        )
    except Exception as e:
        print(f"Ingestion Error: {e}")
        # Re-raise HTTPException to ensure the client receives a correct error status
        if isinstance(e, HTTPException):
            raise e
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
        # This catches errors not specifically handled above, like failed retrieval
        print(f"Query Error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

# To run the server: uvicorn rag_backend_server:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)