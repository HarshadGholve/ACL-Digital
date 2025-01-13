## **Content**
- SQL Code Generation using PyTorch
- Dynamic Document Q&A App using RAG Pipeline

---

# **SQL Code Generation using PyTorch** 🧠💻  
This project focuses on generating SQL queries from natural language inputs using a Transformer-based model built with **PyTorch**. The solution leverages **PostgreSQL** for database interaction and follows a structured approach to data processing, model training, and prediction.

## **📂 Project Structure**  
```
SQL_Code_Generation/
│
├── data/                     # Dataset files (excluded from GitHub)
├── src/                      # Source code
│   ├── __init__.py           
│   ├── data_processing.py    # Data loading and preprocessing
│   ├── model.py              # Transformer model implementation
│   ├── train.py              # Model training script
│   ├── predict.py            # Query prediction script
│   └── utils.py              # Helper functions
└── .gitignore                # Ignored large files (e.g., datasets)
```

## **🚀 How to Run**  
### **1. Clone the Repository**  
```bash
git clone https://github.com/HarshadGholve/ACL-Digital.git
cd ACL-Digital/SQL_Code_Generation
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Train the Model**  
```bash
python src/train.py
```

### **4. Make Predictions**  
Provide a natural language question to generate SQL queries:  
```bash
python src/predict.py "<your question here>"
```

### Example:  
```bash
python src/predict.py "List all restaurants with a rating above 4."
```

## **⚙️ Requirements**  
- Python 3.10+  
- PyTorch  
- PostgreSQL  
- NLTK  
- Transformers  

## **📦 Dataset**  
The datasets are not included in the repository due to size constraints. You can download the datasets from **Kaggle**:  
➡️ [Yale University's SPIDER NLP Dataset](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset/data)  

Place the downloaded datasets in the following path:  
```
data/raw/
```

---

# **Dynamic Document Q&A App using RAG Pipeline** 🤖📚  
A **Retrieval-Augmented Generation (RAG)** project powered by **Streamlit**, **ChromaDB**, and **GroqAPI**, enabling users to ask questions based on uploaded documents or web links. The app retrieves relevant context and generates precise answers using **Llama3-8b-8192** and **HuggingFace text embeddings**.

## **📂 Project Structure**  
```
LangChain_RAG/
│
├── .env                    # Environment variables (API keys, configs)
├── app.py                  # Main Streamlit application
├── utils.py                # Utility functions for embedding and file handling
├── vector_store.py         # ChromaDB setup and vector store interaction
├── retrieval.py            # Document retrieval logic
├── generation.py           # LLM-based answer generation
├── components.py           # Streamlit UI components
└── requirements.txt        # Dependencies
```

## **🚀 How to Run the Project**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/HarshadGholve/ACL-Digital.git
cd ACL-Digital/LangChain_RAG
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Add API Keys**  
Create a `.env` file in the root directory and add your keys:  
```env
API_KEY=<your_groq_api_key>
LANGCHAIN_TRACING_V2=<your_langsmith_token>
```

### **4. Run the App**  
```bash
streamlit run app.py
```

## **🌟 Features**  
- **Upload Documents**: Users can upload PDF, TXT, or CSV files as reference documents.  
- **Web Link Support**: Provide a URL to fetch relevant content for answering questions.  
- **Real-Time Q&A**: The app retrieves context from documents and generates concise answers using LLM.  
- **Persistent Vector Store**: Utilizes **ChromaDB** with DuckDB+Parquet for persistent storage.  
- **Interactive UI**: Built with **Streamlit** for a user-friendly experience.

## **📦 Requirements**  
- Python 3.10+  
- Streamlit  
- ChromaDB  
- GroqAPI  
- HuggingFace Transformers  
- Sentence-Transformers  

## **💻 Technologies Used**  
| **Component**        | **Technology**             | **Model Used**              |
|----------------------|----------------------------|-----------------------------|
| Frontend             | Streamlit                  | -                           |
| Vector Store         | ChromaDB                   | DuckDB+Parquet              |
| Chat Model           | GroqAPI                    | Llama3-8b-8192              |
| Embedding Model      | HuggingFace Transformers   | text-embedding-3-large      |
| Workflow Orchestration | LangGraph                | -                           |
| Tracing & Observability | LangSmith               | -                           |


💡 **Future Enhancements**  
- Multi-file uploads for complex queries.  
- Add caching for faster responses.  
- Human-in-the-loop feedback mechanism.

## **👨‍💻 Author**  
**Harshad Gholve**  
- [GitHub](https://github.com/HarshadGholve)  
- [LinkedIn](https://www.linkedin.com/in/harshad-gholve/)  
