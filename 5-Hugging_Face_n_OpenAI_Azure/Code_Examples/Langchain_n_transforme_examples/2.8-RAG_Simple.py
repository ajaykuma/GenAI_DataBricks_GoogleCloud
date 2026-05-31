'''
Note** All other dependencies should already be installed

!pip install streamlit
--Then run code 
streamlit run RAG_Simple.py

'''
import streamlit as st
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# -----------------------------
# Build Retriever (cached)
# -----------------------------
@st.cache_resource
def build_retriever(file):
    df = pd.read_excel(file)

    docs = [Document(page_content=str(row[0])) for row in df.values]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# Load LLM (cached)
# -----------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)


# -----------------------------
# Prompt Template
# -----------------------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")


# -----------------------------
# Helper to format docs
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("RAG App (Latest LangChain)")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    retriever = build_retriever(uploaded_file)
    llm = load_llm()

    # -----------------------------
    # LCEL RAG Chain
    # -----------------------------
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    user_query = st.text_input("Ask a question:")

    if user_query:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(user_query)

        st.subheader("Answer")
        st.write(answer)

        # -----------------------------
        # Show Sources (optional)
        # -----------------------------
        docs = retriever.invoke(user_query)

        st.subheader("Sources")
        for doc in docs:
            st.write("-", doc.page_content)

else:
    st.info("Upload an Excel file to begin.")
