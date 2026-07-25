#RAG_Simplev2.py -- similarity_search_with_score()
#Other options
#RAG_Simplev1.py -- similarity_search() - Basic retrieval
#RAG_Simplev3.py -- similarity_search_with_score()+ score threshold - Ignore weak matches and reduce hallucinations

#Here
#User Question > VectorStore > similarity_search_with_score() > Sort by Score
#> Retrieved Documents > format_docs() > Prompt > Azure OpenAI > Answer
"""
===============================================================================
RAG_Simplev2.py
Version 2
-----------------
This version builds on v1 by switching from similarity_search() to
similarity_search_with_score(), which returns each retrieved chunk
along with its FAISS distance score (lower score = better match).

This lets us:
    - Inspect retrieval quality before trusting the answer
    - Sort/rank chunks explicitly by relevance
    - See Best / Worst / Average score across retrieved chunks
    - Spot weak or irrelevant matches (useful groundwork for v3's
      score-threshold filtering)
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
    page_title="RAG App - Version 2",
    layout="wide"
)

st.title("RAG App (Version 2 - Using similarity_search())")

st.markdown("""
This performs document retrieval directly from FAISS vector store
using **similarity_search_with_score()** instead of LangChain's Retriever abstraction.

Unlike v1, this version also surfaces the **distance score** for every
retrieved chunk, so you can:
-   see how confident/relevant each retrieved chunk actually is
-   compare Best, Worst, and Average scores across the top-k results
-   inspect ranked chunks (sorted by ascending distance) in the expanders below
-   use this insight to catch weak matches before they cause hallucinated answers

Note: Retrieval quality is visible here, but nothing is filtered out yet —
that's handled in v3 with a score threshold.
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
        # retrieve the top-k most similar document chunks, each paired
        # with its FAISS distance score.
        # Lower score = Better match (more similar to the query).
        # -------------------------------------------------------------
            docs_with_scores = vectorstore.similarity_search_with_score(
            query=user_query,
            k=5
            )

        # -------------------------------------------------------------
        # Step 2
        # Sort results by ascending distance score, so the strongest
        # (most relevant) matches appear first.
        # -------------------------------------------------------------
            docs_with_scores = sorted(
            docs_with_scores,
            key=lambda x: x[1]
            )

        # -------------------------------------------------------------
        # Step 3
        # Extract only the Document objects (drop the scores) so the
        # retrieved chunks can be combined into a single context string
        # for the prompt.
        # -------------------------------------------------------------
            retrieved_docs = [
            doc
            for doc, score in docs_with_scores
            ]

            context = format_docs(retrieved_docs)

        # -------------------------------------------------------------
        # Step 4
        # Send the retrieved context and question to the LLM to
        # generate the final answer.
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

        st.write(f"Total Chunks Retrieved : **{len(docs_with_scores)}**")

        scores = [score for _, score in docs_with_scores]

        st.write(f"Best Score : **{min(scores):.4f}**")
        st.write(f"Worst Score : **{max(scores):.4f}**")
        st.write(f"Average Score : **{sum(scores)/len(scores):.4f}**")

        for rank, (doc, score) in enumerate(docs_with_scores, start=1):

            with st.expander(f"Rank {rank}"):

                st.write(f"**Distance Score:** {score:.6f}")

                st.write(f"**Characters:** {len(doc.page_content)}")

                st.markdown("---")

                st.write(doc.page_content)

    else:

        st.info(
            "Please upload an Excel file to begin."
        )
