import os
import pandas as pd
import streamlit as st
#from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA

# -----------------------------
# Load Excel Data and Build RAG
# -----------------------------
@st.cache_resource
def build_retriever(file_path: str):
    df = pd.read_excel(file_path)
    docs = [Document(page_content=str(row[0])) for row in df.values]

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # embeddings = AzureOpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     azure_endpoint="https://singalkaay-7416-resource.cognitiveservices.azure.com/",           # your endpoint
    #     api_version="2023-12-01-preview",
    #     api_key="mykey",
    # )

    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("RAG Demo with Azure OpenAI")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    retriever = build_retriever(uploaded_file)
    llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",         # deployment name from Azure portal
    azure_endpoint="https://singalkaay-7416-resource.cognitiveservices.azure.com/",           # your endpoint
    api_version="2023-12-01-preview",
    api_key="mykey",
    temperature=0.0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    user_query = st.text_input("Ask a question about your data:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})

        st.subheader("Answer:")
        st.write(result["result"])

        st.subheader("Sources:")
        for doc in result["source_documents"]:
            st.write("-", doc.page_content)
else:
    st.info("Please upload an Excel file to get started.")
