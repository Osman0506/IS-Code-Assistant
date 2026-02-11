# IS-Code-Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?logo=google&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=chainlink&logoColor=white)

**CivilCode AI** is a Retrieval-Augmented Generation (RAG) application designed to help Civil Engineers instantly query massive Indian Standard (IS) Codes. Instead of manually searching through hundreds of pages of PDF regulations, engineers can ask natural language questions and receive precise, cited answers.

---

##  Key Features

* **Semantic Search:** Uses vector embeddings to understand the *meaning* of a query, not just keyword matching.
* **Source Citations:** Every answer includes exact excerpts from the IS Code to ensure engineering accuracy and verification.
* **Multi-Document Support:** Capable of indexing and querying multiple IS codes simultaneously (e.g., IS 456 and IS 800 together).
* **Chat History:** Maintains context of the conversation for follow-up questions.
* **Secure API Handling:** Uses local environment variables to keep API keys safe.

##  Tech Stack

* **Frontend:** Streamlit (for rapid UI development)
* **LLM:** Google Gemini-1.5-Flash (via Google GenAI API)
* **Embeddings:** Google Generative AI Embeddings
* **Vector Store:** FAISS (Facebook AI Similarity Search) for efficient local similarity search
* **Orchestration:** LangChain Community (for document loading and splitting)
* **PDF Processing:** PyPDF2

---

##  Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/CivilCode-AI.git](https://github.com/YOUR-USERNAME/CivilCode-AI.git)
cd CivilCode-AI
