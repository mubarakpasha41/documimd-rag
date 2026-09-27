import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="DocuMind RAG", page_icon="📄")
st.title("📄 DocuMind RAG")
st.caption("Upload a PDF and ask questions based only on its contents.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY was not found. Add it to your .env file.")
    st.stop()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    new_file_uploaded = uploaded_file.name != st.session_state.document_name

    if new_file_uploaded:
        with st.spinner("Reading, splitting, and indexing the PDF..."):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_path)
                pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(pages)

                embeddings = OpenAIEmbeddings()
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                st.session_state.document_name = uploaded_file.name

                st.success(
                    f"Indexed {len(chunks)} text chunks from "
                    f"`{uploaded_file.name}`."
                )
            except Exception as error:
                st.error(f"Could not process the PDF: {error}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

question = st.chat_input("Ask a question about the uploaded PDF")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                retrieved_docs = st.session_state.vector_store.similarity_search(
                    question,
                    k=4
                )

                context = "\n\n".join(
                    [
                        f"[Page {doc.metadata.get('page', 0) + 1}]\n"
                        f"{doc.page_content}"
                        for doc in retrieved_docs
                    ]
                )

                prompt = f"""
You are a helpful document assistant.

Answer the question using only the context below.
If the answer cannot be found in the context, say exactly:
"I could not find that information in the uploaded document."

Mention the relevant page number or page numbers in your answer.

Context:
{context}

Question:
{question}
"""

                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                )

                response = llm.invoke(prompt)
                st.write(response.content)

                with st.expander("Retrieved source passages"):
                    for doc in retrieved_docs:
                        page_number = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Page {page_number}**")
                        st.write(doc.page_content[:700] + "...")