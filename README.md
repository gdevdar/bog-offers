# BoG Offers RAG System

A Retrieval-Augmented Generation (RAG) system for interacting with Bank of Georgia's offers using a conversational web interface.

## Features
- **RAG-powered chatbot**: Ask questions about BoG offers and get relevant, structured answers.
- **Web interface**: Modern, responsive chat UI served by Flask.


---

## Quick Start

### 1. Clone the repository
```
git clone git@github.com:gdevdar/bog-offers.git
or
git clone https://github.com/gdevdar/bog-offers.git
cd bog-offers
```

### 2. Create and activate a virtual environment (optional but recommended)
```
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up your `.env` file
Create a file named `.env` in the project root with your Gemini API key:
```
API_KEY=your-gemini-api-key-here
```
**Do NOT share or commit your `.env` file!**

### 5. Prepare the data (if needed)
If you need to update or embed new offers, run:
```
python full_system.py
```
First prompt will allow you to fetch and embed data. You can skip the second prompt as it is for chatting with RAG in the terminal.

### 6. Run the web app
```
python app.py
```
Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser to chat with the RAG model.

---

## Project Structure
```
bog-offers/
├── app.py
├── full_system.py
├── index.html
├── requirements.txt
├── README.md
├── .env                # (Not committed) Your API key
├── .gitignore
│
├── generated_files/
│   ├── chunks_metadata.pkl
│   ├── faiss_index.bin
│   └── offers.json
│
├── rag_scripts/
│   ├── offer_recommender.py
│   ├── embed_data.py
│   ├── fetch_data.py
│   └── __init__.py
│   └── __pycache__/         # (Not committed) Python bytecode cache
│
├── .venv/                   # (Not committed) Python virtual environment
│   ├── Scripts/
│   ├── Lib/
│   ├── Include/
│   ├── pyvenv.cfg
│   └── .gitignore
│
```
- `.env`, `.venv/`, and `__pycache__/` are not committed to version control (see `.gitignore`).
- All data files are stored in `generated_files/`.
- All RAG logic and scripts are in `rag_scripts/`.


