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

from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")

#embeddings = OpenAIEmbeddings()
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
)

# Seed with sample patient summaries
INITIAL_DOCS = [
    Document(
        page_content="Patient P001 John Smith, 70yo male with CKD Stage 3 and hypertension. "
                     "On Lisinopril and Amlodipine. Allergic to Penicillin. Last seen Nov 2024.",
        metadata={"patient_id": "P001"}
    ),
    Document(
        page_content="CKD management includes dietary protein restriction, BP control below 130/80, "
                     "ACE inhibitors as first-line therapy, and quarterly creatinine monitoring.",
        metadata={"type": "clinical_guideline", "condition": "CKD"}
    )
]

vector_store = FAISS.from_documents(INITIAL_DOCS, embeddings)

def retrieve_context(query: str, k: int = 2) -> str:
    """Retrieve relevant context from vector store."""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

def add_to_memory(text: str, metadata: dict = {}):
    """Add new document to vector memory."""
    doc = Document(page_content=text, metadata=metadata)
    vector_store.add_documents([doc])
