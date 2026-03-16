# FAISS vector memory with PDF + JSON + seed document support
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Embeddings ────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "../data/faiss_index")

# ── Source 1: Seed documents ──────────────────────────────────────
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
        page_content="Patient P003 Robert Lee, 58yo male with Coronary Artery Disease. "
                     "On Atorvastatin and Aspirin. Allergic to contrast dye.",
        metadata={"source": "hardcoded", "patient_id": "P003"}
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

# ── Source 2: patients.json ───────────────────────────────────────
PATIENTS_JSON = os.path.join(os.path.dirname(__file__), "../data/patients.json")

def load_patient_json_as_docs() -> list:
    if not os.path.exists(PATIENTS_JSON):
        print("patients.json not found, skipping.")
        return []
    with open(PATIENTS_JSON, "r") as f:
        patients = json.load(f)
    docs = []
    for pid, data in patients.items():
        summary = (
            f"Patient ID: {pid}\n"
            f"Name: {data.get('name')}, Age: {data.get('age')}, "
            f"Gender: {data.get('gender')}\n"
            f"Conditions: {', '.join(data.get('conditions', []))}\n"
            f"Medications: {', '.join(data.get('medications', []))}\n"
            f"Allergies: {', '.join(data.get('allergies', []))}\n"
            f"Last Visit: {data.get('last_visit')}\n"
            f"Doctor: {data.get('doctor')}\n"
        )
        docs.append(Document(
            page_content=summary,
            metadata={"source": "patients.json", "patient_id": pid,
                      "type": "patient_summary"}
        ))
        for visit in data.get("visit_history", []):
            docs.append(Document(
                page_content=(
                    f"Patient {data.get('name')} ({pid}) visit on {visit['date']}: "
                    f"Reason: {visit['reason']}. Notes: {visit['notes']}"
                ),
                metadata={"source": "patients.json", "patient_id": pid,
                          "type": "visit_history"}
            ))
        labs = data.get("lab_results", {})
        if labs:
            docs.append(Document(
                page_content=f"Lab results for {data.get('name')} ({pid}): " +
                             ", ".join([f"{k}: {v}" for k, v in labs.items()]),
                metadata={"source": "patients.json", "patient_id": pid,
                          "type": "lab_results"}
            ))
        # Load any saved notes
        for note in data.get("notes", []):
            docs.append(Document(
                page_content=(
                    f"Clinical note for {data.get('name')} ({pid}) "
                    f"on {note.get('date')} {note.get('time')}: {note.get('note')}"
                ),
                metadata={"source": "patients.json", "patient_id": pid,
                          "type": "clinical_note"}
            ))
    print(f"Loaded patients.json → {len(docs)} documents")
    return docs

# ── Source 3: PDFs from data/docs/ ───────────────────────────────
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "../data/docs")

def load_pdf_documents() -> list:
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print("Created data/docs/ — add PDFs here for RAG ingestion.")
        return []
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in data/docs/")
        return []
    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(DOCS_FOLDER, pdf_file)
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_file
                page.metadata["type"] = "pdf_document"
            chunks = text_splitter.split_documents(pages)
            all_docs.extend(chunks)
            print(f"Loaded PDF: {pdf_file} → {len(chunks)} chunks")
        except Exception as e:
            print(f"Failed to load {pdf_file}: {e}")
    return all_docs

# ── Build vector store from all sources ──────────────────────────
def build_vector_store() -> FAISS:
    all_docs = []
    all_docs.extend(SEED_DOCS)
    all_docs.extend(load_patient_json_as_docs())
    all_docs.extend(load_pdf_documents())
    print(f"\n Total documents in vector store: {len(all_docs)}")
    return FAISS.from_documents(all_docs, embeddings)

def save_vector_store():
    """Persist FAISS index to disk so notes survive restarts."""
    vector_store.save_local(FAISS_INDEX_PATH)
    print(" FAISS index saved to disk")

def load_vector_store() -> FAISS:
    """Load existing FAISS index if available, else build fresh."""
    if os.path.exists(FAISS_INDEX_PATH):
        print(" Loading existing FAISS index from disk")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    print(" Building fresh FAISS index...")
    return build_vector_store()

# ── Initialize on import ──────────────────────────────────────────
vector_store = load_vector_store()

# ── Public functions ──────────────────────────────────────────────
def retrieve_context(query: str, k: int = 3) -> str:
    docs = vector_store.similarity_search(query, k=k)
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)

def add_to_memory(text: str, metadata: dict = {}):
    """Add text to FAISS and persist to disk."""
    doc = Document(page_content=text, metadata=metadata)
    vector_store.add_documents([doc])
    save_vector_store()
    print(f" Added to memory: {text[:60]}...")

def ingest_single_pdf(pdf_path: str) -> int:
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        filename = os.path.basename(pdf_path)
        for page in pages:
            page.metadata["source"] = filename
            page.metadata["type"] = "pdf_document"
        chunks = text_splitter.split_documents(pages)
        vector_store.add_documents(chunks)
        save_vector_store()
        print(f" Ingested {filename} → {len(chunks)} chunks")
        return len(chunks)
    except Exception as e:
        print(f" Failed to ingest PDF: {e}")
        return 0
