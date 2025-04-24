import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

st.title("Text Book RAG")

if st.sidebar.button("Load Pdf from Directory"):
    loader = DirectoryLoader("./data/sixth/science", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = VertexAIEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
    st.session_state["vectordb"] = vectordb

query = st.text_input("Enter your query")
if query:
    vectordb = st.session_state["vectordb"]
    retriever = vectordb.as_retriever()
    llm = ChatVertexAI(model_name="gemini-pro")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant. Given the following context, answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    answer = chain.run({"context": context, "question": query})
    #st.write(answer)
    # Output
    st.subheader("Generated Answer:")
    st.write(answer)


