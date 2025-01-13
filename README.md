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
The datasets are not included in the repository due to size constraints. You can download the datasets and place them in the following path:  
`data/raw/`

## **👨‍💻 Author**  
**Harshad Gholve**  
- [GitHub](https://github.com/HarshadGholve)  
- [LinkedIn](https://www.linkedin.com/in/harshad-gholve/)  

---

Feel free to tweak it as needed! 😊
