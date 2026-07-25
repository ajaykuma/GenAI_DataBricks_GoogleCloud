#RAG_Simplev1.py -- similarity_search()
#Other options
#RAG_Simplev2.py -- similarity_search_with_score() - Inspect retrieval quality using similarity scores
#RAG_Simplev3.py -- similarity_search_with_score()+ score threshold - Ignore weak matches and reduce hallucinations

#Here
#User Question > VectorStore > similarity_search() > Retrieved Documents > format_docs() > Prompt
#> Azure OpenAI > Answer

"""
===============================================================================
RAG_Simplev1.py
Version 1 
-----------------
This version removes LangChain's Retriever abstraction and performs retrieval
directly using:
    vectorstore.similarity_search(question, k=5)
Now we can see what the Retriever is doing internally.
===============================================================================
"""

import os
import torch
import streamlit as st
import pandas as pd

from dotenv import load_dotenv

#If swtching to use HF models, we can uncomment this & make changes in load_llm as shown in v0
#from transformers import (pipeline,AutoTokenizer,AutoModelForSeq2SeqLM,)
#from langchain_huggingface import HuggingFacePipeline

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

# =============================================================================
# Azure OpenAI Configuration
# =============================================================================

load_dotenv()

api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_API_VERSION")
deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

# =============================================================================
# Build FAISS Vector Store
# =============================================================================

@st.cache_resource
def build_vectorstore(file):
    """
    Reads the uploaded Excel file, converts each row into a LangChain Document,
    splits long text into chunks, creates embeddings, and stores them in FAISS.

    Returns
    -------
    FAISS
        FAISS vector store containing embedded document chunks.
    """

    df = pd.read_excel(file)

    docs = [
        Document(page_content=str(row[0]))
        for row in df.values
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings,
    )

    return vectorstore


# =============================================================================
# Load Azure OpenAI LLM
# =============================================================================

@st.cache_resource
def load_llm():
    """
    Loads the Azure OpenAI chat model.
    Returns AzureChatOpenAI
    """

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment,
    )

    return llm


# =============================================================================
# Prompt Template
# =============================================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question ONLY using the context below.

If the answer is not in the context,
reply exactly:

"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)

# =============================================================================
# Helper Function
# =============================================================================

def format_docs(docs):
    """
    Combines multiple retrieved documents into a single string.

    Parameters
    ----------
    docs : list[Document]
    Returns - str
    """

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )
# =============================================================================
# Streamlit User Interface
# =============================================================================

st.set_page_config(
    page_title="RAG App - Version 1",
    layout="wide"
)

st.title("RAG App (Version 1 - Using similarity_search())")

st.markdown("""
This performs document retrieval directly from FAISS vector store
using **similarity_search()** instead of LangChain's Retriever abstraction.
Note: Since we use, Answer ONLY using the context, it 
-   has low hallucination
-   good for enterprise search
-   good for internal company documents
-   doesn't answer general knowledge
""")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

# =============================================================================
# Main Application
# =============================================================================

if uploaded_file:

    # -------------------------------------------------------------------------
    # Build Vector Store
    # -------------------------------------------------------------------------
    with st.spinner("Building Vector Store..."):
        vectorstore = build_vectorstore(uploaded_file)

    # -------------------------------------------------------------------------
    # Load LLM
    # -------------------------------------------------------------------------
    llm = load_llm()

    # -------------------------------------------------------------------------
    # Build Question Answering Chain
    #
    # Retrieval is performed manually using similarity_search().
    # Only prompting and LLM invocation use LCEL.
    # -------------------------------------------------------------------------
    qa_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # -------------------------------------------------------------------------
    # User Question
    # -------------------------------------------------------------------------
    user_query = st.text_input(
        "Ask a question:"
    )

    if user_query:

        with st.spinner("Searching documents..."):

            # -------------------------------------------------------------
            # Step 1
            # Convert the user question into an embedding internally and
            # retrieve the top-k most similar document chunks.
            # -------------------------------------------------------------
            retrieved_docs = vectorstore.similarity_search(
                query=user_query,
                k=5
            )

            # -------------------------------------------------------------
            # Step 2
            # Convert retrieved documents into one context string.
            # -------------------------------------------------------------
            context = format_docs(retrieved_docs)

            # -------------------------------------------------------------
            # Step 3
            # Send the retrieved context and question to the LLM.
            # -------------------------------------------------------------
            answer = qa_chain.invoke(
                {
                    "context": context,
                    "question": user_query,
                }
            )

        # -----------------------------------------------------------------
        # Display Answer
        # -----------------------------------------------------------------
        st.subheader("Answer")

        st.write(answer)

        # -----------------------------------------------------------------
        # Display Retrieved Documents
        # -----------------------------------------------------------------
        st.subheader("Retrieved Document Chunks")

        st.write(f"Total Chunks Retrieved : **{len(retrieved_docs)}**")

        for i, doc in enumerate(retrieved_docs, start=1):

            with st.expander(f"Chunk {i}"):

                st.write(doc.page_content)

                st.caption(
                    f"Characters : {len(doc.page_content)}"
                )

else:

    st.info(
        "Please upload an Excel file to begin."
    )
