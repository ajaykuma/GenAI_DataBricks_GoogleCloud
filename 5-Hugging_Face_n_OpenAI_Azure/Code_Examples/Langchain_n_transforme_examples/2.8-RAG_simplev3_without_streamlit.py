# =============================================================================
# 2.8-RAG_simplev3_without_streamlit.py
# run > python 2.8-RAG_simplev3_without_streamlit.py
#
# Version 3 (Command Line)
#
# This version removes Streamlit completely and runs from the command line.
#
# Retrieval Pipeline
#
# User Question
#       ↓
# similarity_search_with_score()
#       ↓
# Sort by Distance Score
#       ↓
# Apply Threshold
#       ↓
# Build Context
#       ↓
# Azure OpenAI
#       ↓
# Answer
# =============================================================================

import os
import pandas as pd

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import AzureChatOpenAI


# =============================================================================
# Configuration
# =============================================================================

# Change this to your Excel file location.
EXCEL_FILE = "E:\\GitContent\\GenAI_DataBricks_GoogleCloud\\GenAI_DataBricks_GoogleCloud\\3-Datasets\\Scenario.xlsx"

# Number of chunks to retrieve.
TOP_K = 5

# Default FAISS distance threshold.
# Lower value = stricter retrieval.
DEFAULT_THRESHOLD = 1.0


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

def build_vectorstore(file_path):
    """
    Read the Excel file, split documents into chunks,
    generate embeddings and build a FAISS vector store.
    """

    print("\nBuilding vector store...")

    df = pd.read_excel(file_path)

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

    print(f"Loaded {len(split_docs)} document chunks.")

    return vectorstore


# =============================================================================
# Load Azure OpenAI
# =============================================================================

def load_llm():

    print("Loading Azure OpenAI...")

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment,
    )

    return llm


# =============================================================================
# Prompt
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
# Helper
# =============================================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 70)
    print("RAG Version 3 (Command Line)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Check Excel file
    # -------------------------------------------------------------------------
    if not os.path.exists(EXCEL_FILE):
        print(f"\nERROR: Excel file not found:\n{EXCEL_FILE}")
        return

    print(f"\nExcel File : {EXCEL_FILE}")

    # -------------------------------------------------------------------------
    # Read score threshold from user
    # -------------------------------------------------------------------------
    threshold_input = input(
        f"\nScore Threshold "
        f"(Press Enter for default {DEFAULT_THRESHOLD}) : "
    ).strip()

    if threshold_input:
        score_threshold = float(threshold_input)
    else:
        score_threshold = DEFAULT_THRESHOLD

    print(f"\nUsing Threshold : {score_threshold}")

    # -------------------------------------------------------------------------
    # Build vector store
    # -------------------------------------------------------------------------
    vectorstore = build_vectorstore(EXCEL_FILE)

    # -------------------------------------------------------------------------
    # Load LLM
    # -------------------------------------------------------------------------
    llm = load_llm()

    qa_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    print("\nSystem Ready.")
    print("Type 'exit' or 'quit' to stop.\n")

    # -------------------------------------------------------------------------
    # Chat Loop
    # -------------------------------------------------------------------------
    while True:

        user_query = input("Question > ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("\nGoodbye.")
            break

        if not user_query:
            continue

        print("\nSearching...")

        # -------------------------------------------------------------
        # Retrieve Top-K Documents with Scores
        # -------------------------------------------------------------
        docs_with_scores = vectorstore.similarity_search_with_score(
            query=user_query,
            k=TOP_K
        )

        # -------------------------------------------------------------
        # Sort by Score
        # -------------------------------------------------------------
        docs_with_scores = sorted(
            docs_with_scores,
            key=lambda x: x[1]
        )

        # -------------------------------------------------------------
        # Apply Threshold
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
        # Build Context
        # -------------------------------------------------------------
        retrieved_docs = [
            doc
            for doc, score in passed_docs
        ]

        context = format_docs(retrieved_docs)

        # -------------------------------------------------------------
        # Generate Answer
        # -------------------------------------------------------------
        answer = qa_chain.invoke(
            {
                "context": context if context else "No relevant context found.",
                "question": user_query,
            }
        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

        # -------------------------------------------------------------
        # Retrieval Statistics
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("RETRIEVAL SUMMARY")
        print("=" * 70)

        print(f"Chunks Retrieved : {len(docs_with_scores)}")
        print(f"Chunks Passed    : {len(passed_docs)}")
        print(f"Chunks Discarded : {len(discarded_docs)}")

        if docs_with_scores:

            scores = [
                score
                for _, score in docs_with_scores
            ]

            print(f"\nBest Score    : {min(scores):.6f}")
            print(f"Worst Score   : {max(scores):.6f}")
            print(f"Average Score : {sum(scores)/len(scores):.6f}")

        # -------------------------------------------------------------
        # Passed Chunks
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PASSED CHUNKS")
        print("=" * 70)

        if passed_docs:

            for rank, (doc, score) in enumerate(
                passed_docs,
                start=1
            ):

                print(f"\nRank {rank}")
                print(f"Score      : {score:.6f}")
                print(f"Characters : {len(doc.page_content)}")
                print("-" * 70)
                print(doc.page_content)

        else:

            print("\nNo chunks passed the threshold.")

        # -------------------------------------------------------------
        # Discarded Chunks
        # -------------------------------------------------------------
        if discarded_docs:

            print("\n" + "=" * 70)
            print("DISCARDED CHUNKS")
            print("=" * 70)

            for rank, (doc, score) in enumerate(
                discarded_docs,
                start=1
            ):

                print(f"\nDiscarded {rank}")
                print(f"Score      : {score:.6f}")
                print(f"Characters : {len(doc.page_content)}")
                print("-" * 70)
                print(doc.page_content)

        print("\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
