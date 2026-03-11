 # FAISS vector memory
#RAG component — storing and retrieving patient summaries.
#The value for AZURE_EMBEDDING_DEPLOYMENT depends 
#on what you deployed in your Azure OpenAI resource. Common options are:

'''
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002   # most common, older
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small   # newer, cheaper
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large   # newest, most accurate
'''

import os
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

# ── Embeddings Client ─────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
)

# ── Text Splitter ─────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# ─────────────────────────────────────────────────────────────────
# SOURCE 1: Hardcoded seed documents (always loaded)
# ─────────────────────────────────────────────────────────────────
SEED_DOCS = [
    Document(
        page_content="Patient P001 John Smith, 70yo male with CKD Stage 3 and hypertension. "
                     "On Lisinopril and Amlodipine. Allergic to Penicillin. Last seen Nov 2024.",
        metadata={"source": "hardcoded", "patient_id": "P001"}
    ),
    Document(
        page_content="Patient P002 Sarah Johnson, 45yo female with Asthma and Hypothyroidism. "
                     "On Albuterol inhaler and Levothyroxine. Allergic to Aspirin.",
        metadata={"source": "hardcoded", "patient_id": "P002"}
    ),
    Document(
        page_content="CKD management includes dietary protein restriction, BP control below 130/80, "
                     "ACE inhibitors as first-line therapy, and quarterly creatinine monitoring.",
        metadata={"source": "hardcoded", "type": "clinical_guideline", "condition": "CKD"}
    ),
    Document(
        page_content="Asthma management includes inhaled corticosteroids as first-line controller therapy, "
                     "short-acting beta agonists for rescue, and regular peak flow monitoring.",
        metadata={"source": "hardcoded", "type": "clinical_guideline", "condition": "Asthma"}
    ),
]

# ─────────────────────────────────────────────────────────────────
# SOURCE 2: Load PDFs from a folder
# Place patient PDFs, clinical guidelines, lab reports in /data/docs/
# ─────────────────────────────────────────────────────────────────
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "../data/docs")

def load_pdf_documents() -> list:
    """Load and chunk all PDFs found in data/docs/ folder."""
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(" Created data/docs/ folder — add PDFs here for RAG ingestion.")
        return []

    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print(" No PDFs found in data/docs/")
        return []

    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(DOCS_FOLDER, pdf_file)
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            # Tag each chunk with the source filename
            for page in pages:
                page.metadata["source"] = pdf_file
                page.metadata["type"] = "pdf_document"
            chunks = text_splitter.split_documents(pages)
            all_docs.extend(chunks)
            print(f" Loaded PDF: {pdf_file} → {len(chunks)} chunks")
        except Exception as e:
            print(f" Failed to load {pdf_file}: {e}")

    return all_docs

# ─────────────────────────────────────────────────────────────────
# SOURCE 3: Load from patients.json as documents
# ─────────────────────────────────────────────────────────────────
PATIENTS_JSON = os.path.join(os.path.dirname(__file__), "../data/patients.json")

def load_patient_json_as_docs() -> list:
    """Convert patients.json entries into searchable documents."""
    if not os.path.exists(PATIENTS_JSON):
        return []

    import json
    with open(PATIENTS_JSON, "r") as f:
        patients = json.load(f)

    docs = []
    for pid, data in patients.items():
        # Build a readable summary per patient
        summary = (
            f"Patient ID: {pid}\n"
            f"Name: {data.get('name')}, Age: {data.get('age')}, Gender: {data.get('gender')}\n"
            f"Conditions: {', '.join(data.get('conditions', []))}\n"
            f"Medications: {', '.join(data.get('medications', []))}\n"
            f"Allergies: {', '.join(data.get('allergies', []))}\n"
            f"Last Visit: {data.get('last_visit')}\n"
            f"Doctor: {data.get('doctor')}\n"
        )
        # Add visit history as separate chunks
        for visit in data.get("visit_history", []):
            visit_text = (
                f"Patient {data.get('name')} ({pid}) visit on {visit['date']}: "
                f"Reason: {visit['reason']}. Notes: {visit['notes']}"
            )
            docs.append(Document(
                page_content=visit_text,
                metadata={"source": "patients.json", "patient_id": pid, "type": "visit_history"}
            ))
        # Add lab results
        labs = data.get("lab_results", {})
        if labs:
            lab_text = f"Lab results for {data.get('name')} ({pid}): " + \
                       ", ".join([f"{k}: {v}" for k, v in labs.items()])
            docs.append(Document(
                page_content=lab_text,
                metadata={"source": "patients.json", "patient_id": pid, "type": "lab_results"}
            ))
        docs.append(Document(
            page_content=summary,
            metadata={"source": "patients.json", "patient_id": pid, "type": "patient_summary"}
        ))

    print(f" Loaded patients.json → {len(docs)} documents")
    return docs

# ─────────────────────────────────────────────────────────────────
# BUILD VECTOR STORE — combines all three sources
# ─────────────────────────────────────────────────────────────────
def build_vector_store() -> FAISS:
    all_docs = []
    all_docs.extend(SEED_DOCS)
    all_docs.extend(load_patient_json_as_docs())
    all_docs.extend(load_pdf_documents())

    if not all_docs:
        raise ValueError("No documents loaded. Add PDFs to data/docs/ or check patients.json.")

    print(f"\n Total documents in vector store: {len(all_docs)}")
    return FAISS.from_documents(all_docs, embeddings)

# Initialize on import
vector_store = build_vector_store()

# ─────────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant chunks from vector store."""
    docs = vector_store.similarity_search(query, k=k)
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)

def add_to_memory(text: str, metadata: dict = {}):
    """Dynamically add any new text into the vector store at runtime."""
    doc = Document(page_content=text, metadata=metadata)
    vector_store.add_documents([doc])
    print(f" Added to memory: {text[:60]}...")

def ingest_single_pdf(pdf_path: str):
    """Ingest a single PDF file into the vector store at runtime.
    Useful for uploading new patient reports via the Streamlit UI.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        filename = os.path.basename(pdf_path)
        for page in pages:
            page.metadata["source"] = filename
            page.metadata["type"] = "pdf_document"
        chunks = text_splitter.split_documents(pages)
        vector_store.add_documents(chunks)
        print(f" Ingested {filename} → {len(chunks)} chunks added to memory")
        return len(chunks)
    except Exception as e:
        print(f" Failed to ingest PDF: {e}")
        return 0
