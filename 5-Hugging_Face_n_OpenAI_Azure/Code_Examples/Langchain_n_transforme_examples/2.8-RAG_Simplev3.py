#RAG_Simplev3.py -- similarity_search_with_score() + score threshold
#Other options
#RAG_Simplev1.py -- similarity_search() - Basic retrieval
#RAG_Simplev2.py -- similarity_search_with_score() - Inspect retrieval quality using similarity scores

#Here
#User Question > VectorStore > similarity_search_with_score() > Sort by Score
#> Filter by Score Threshold > Retrieved Documents > format_docs() > Prompt
#> Azure OpenAI > Answer

"""
===============================================================================
RAG_Simplev3.py
Version 3
-----------------
This version builds on v2 by adding a score threshold on top of
similarity_search_with_score(). Chunks whose distance score is worse
than (i.e. greater than) the threshold are discarded before being
passed to the LLM.

This lets us:
    - Ignore weak / irrelevant matches instead of blindly using top-k
    - Reduce hallucinations caused by low-quality context
    - Explicitly answer "I don't know." when no chunk is good enough,
      rather than forcing an answer from poor context
    - Tune the threshold interactively and see how many chunks survive
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
    page_title="RAG App - Version 3",
    layout="wide"
)

st.title("RAG App (Version 3 - Score Threshold Filtering)")

st.markdown("""
This performs document retrieval directly from FAISS vector store
using **similarity_search_with_score()**, then applies a **score threshold**
to discard weak matches before they ever reach the LLM.

Unlike v2 (which only inspects scores), this version:
-   filters out chunks whose distance score is worse than the threshold
-   reduces hallucinations caused by low-quality / irrelevant context
-   can return **zero** chunks if nothing is good enough, forcing the
    assistant to reply "I don't know." instead of guessing
-   lets you tune the threshold live and see how many chunks survive
    vs how many were discarded
""")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

# -------------------------------------------------------------------------
# Score Threshold Control
#
# FAISS returns L2 distance by default: lower score = more similar.
# Chunks with score > threshold are treated as weak matches and dropped.
# -------------------------------------------------------------------------
score_threshold = st.slider(
    "Score Threshold (lower = stricter, fewer chunks kept)",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05,
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
    # Retrieval is performed manually using similarity_search_with_score(),
    # followed by threshold filtering. Only prompting and LLM invocation
    # use LCEL.
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
        # Retrieve the top-k matching document chunks along with
        # their FAISS distance scores.
        # Lower score = Better match.
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
        # Apply the score threshold. Keep only chunks whose distance
        # score is less than or equal to the threshold; everything
        # else is treated as a weak match and discarded.
        # -------------------------------------------------------------
            passed_docs = [
            (doc, score)
            for doc, score in docs_with_scores
            if score <= score_threshold
            ]

            discarded_docs = [
            (doc, score)
            for doc, score in docs_with_scores
            if score > score_threshold
            ]

        # -------------------------------------------------------------
        # Step 4
        # Extract only the Document objects that passed the threshold
        # and build the context string for the prompt.
        # If nothing passed, context is empty and the prompt will
        # instruct the LLM to reply "I don't know."
        # -------------------------------------------------------------
            retrieved_docs = [
            doc
            for doc, score in passed_docs
            ]

            context = format_docs(retrieved_docs)

        # -------------------------------------------------------------
        # Step 5
        # Generate the final answer. If context is empty, the prompt
        # itself enforces the "I don't know." fallback.
        # -------------------------------------------------------------
            answer = qa_chain.invoke(
            {
                "context": context if context else "No relevant context found.",
                "question": user_query,
            }
            )

        # -----------------------------------------------------------------
        # Display Answer
        # -----------------------------------------------------------------
        st.subheader("Answer")

        st.write(answer)

        # -----------------------------------------------------------------
        # Display Retrieval Summary
        # -----------------------------------------------------------------
        st.subheader("Retrieval Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Chunks Retrieved (k)", len(docs_with_scores))
        col2.metric("Chunks Passed Threshold", len(passed_docs))
        col3.metric("Chunks Discarded", len(discarded_docs))

        if docs_with_scores:
            scores = [score for _, score in docs_with_scores]
            st.write(f"Best Score : **{min(scores):.4f}**")
            st.write(f"Worst Score : **{max(scores):.4f}**")
            st.write(f"Average Score : **{sum(scores)/len(scores):.4f}**")

        if not passed_docs:
            st.warning(
                "No chunks passed the score threshold. "
                "Try increasing the threshold or refining your question."
            )

        # -----------------------------------------------------------------
        # Display Retrieved (Passed) Document Chunks
        # -----------------------------------------------------------------
        st.subheader("Retrieved Document Chunks (Passed Threshold)")

        for rank, (doc, score) in enumerate(passed_docs, start=1):

            with st.expander(f"Rank {rank} - Kept"):

                st.write(f"**Distance Score:** {score:.6f}")

                st.write(f"**Characters:** {len(doc.page_content)}")

                st.markdown("---")

                st.write(doc.page_content)

        # -----------------------------------------------------------------
        # Display Discarded Document Chunks
        # -----------------------------------------------------------------
        if discarded_docs:

            st.subheader("Discarded Chunks (Below Threshold)")

            for rank, (doc, score) in enumerate(discarded_docs, start=1):

                with st.expander(f"Discarded {rank}"):

                    st.write(f"**Distance Score:** {score:.6f}")

                    st.write(f"**Characters:** {len(doc.page_content)}")

                    st.markdown("---")

                    st.write(doc.page_content)

else:

    st.info(
        "Please upload an Excel file to begin."
    )
