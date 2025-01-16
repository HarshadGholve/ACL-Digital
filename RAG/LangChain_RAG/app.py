import os
import pandas as pd
import streamlit as st
from utils.document_processing import process_uploaded_file, process_url, split_documents, add_documents_to_store
from utils.vector_store_utils import initialize_vector_store, search_documents
from utils.feedback import store_feedback
from langchain import hub

# Initialize environment variables
API_KEY = os.getenv("API_KEY")
vector_store = initialize_vector_store(API_KEY)

# Load RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# Stats
stats = {"total_documents": 0, "total_questions": 0, "positive_feedback": 0, "negative_feedback": 0}

# App UI
st.title("RAG Pipeline with Analytics")
menu = st.sidebar.selectbox("Menu", ["Upload File", "Enter URL", "Ask a Question", "Analytics"])

if menu == "Upload File":
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
    if uploaded_file:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        documents = process_uploaded_file(file_path)
        chunks = split_documents(documents)
        add_documents_to_store(vector_store, chunks)
        stats["total_documents"] += 1
        st.success(f"Uploaded and processed {len(chunks)} chunks.")

elif menu == "Enter URL":
    url = st.text_input("Enter a URL:")
    if st.button("Process URL"):
        documents = process_url(url)
        chunks = split_documents(documents)
        add_documents_to_store(vector_store, chunks)
        stats["total_documents"] += 1
        st.success(f"Processed URL and added {len(chunks)} chunks.")

elif menu == "Ask a Question":
    question = st.text_input("Your question:")
    if st.button("Get Answer"):
        retrieved_docs = search_documents(vector_store, question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        response = prompt.invoke({"context": context, "question": question})
        answer = response.content
        st.write(f"Answer: {answer}")
        feedback = st.radio("Feedback:", ["👍", "👎"])
        store_feedback(question, answer, feedback)
        stats["total_questions"] += 1
        if feedback == "👍":
            stats["positive_feedback"] += 1
        else:
            stats["negative_feedback"] += 1

elif menu == "Analytics":
    st.metric("Total Documents Uploaded", stats["total_documents"])
    st.metric("Total Questions Asked", stats["total_questions"])
    feedback_chart = pd.DataFrame({
        "Feedback": ["Positive", "Negative"],
        "Count": [stats["positive_feedback"], stats["negative_feedback"]],
    })
    st.bar_chart(feedback_chart.set_index("Feedback"))
