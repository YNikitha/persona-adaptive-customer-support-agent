# 🧠 Persona-Adaptive Customer Support Agent

An intelligent AI-powered customer support system that dynamically adapts responses based on user persona using **LLMs, Retrieval-Augmented Generation (RAG), vector search, and escalation logic**.

---

## 🚀 Objective

This project builds a smart customer support agent that:

- Detects user persona (Technical Expert / Frustrated User / Business Executive)
- Retrieves relevant knowledge using RAG (Retrieval-Augmented Generation)
- Generates adaptive responses based on persona style
- Escalates complex or sensitive issues to human support

---

## 🏗️ System Architecture
User Query
│
▼
Persona Classifier (Gemini)
│
▼
Vector Database (ChromaDB)
│
▼
Semantic Search (Cosine Similarity)
│
▼
Top-K Retrieved Chunks
│
▼
Persona-Based Prompt Engine
│
├── High Confidence → AI Response Generated
└── Low Confidence / Sensitive Issue → Human Escalation (JSON Handoff)

---

## ⚙️ Tech Stack

- Python 3.11+
- Google Gemini API (`google-genai`)
- Streamlit (UI)
- LangChain (Text Splitting)
- ChromaDB / FAISS (Vector Database)
- PyPDF (PDF parsing)
- python-dotenv (Environment variables)

---

## 📁 Project Structure
persona-support-agent/
│
├── data/
│ ├── api_troubleshooting.md
│ ├── billing_policy.txt
│ └── password_reset_guide.pdf
│
├── src/
│ ├── config.py
│ ├── classifier.py
│ ├── rag_pipeline.py
│ ├── generator.py
│ └── escalator.py
│
├── app.py
├── requirements.txt
├── .env
└── README.md

---

## 🔄 How It Works

### 1. Persona Classification
User message is analyzed using Gemini to classify tone and intent into a persona.

### 2. RAG Pipeline
- Documents loaded from `/data`
- Split into chunks using RecursiveCharacterTextSplitter
- Converted into embeddings using `text-embedding-004`
- Stored in ChromaDB

### 3. Semantic Retrieval
User query is converted into embeddings and matched using cosine similarity to fetch top relevant chunks.

### 4. Adaptive Response Generation
Responses are generated based on persona:

- 🧑‍💻 Technical Expert → Deep technical explanation, logs, APIs
- 😡 Frustrated User → Empathetic tone + simple step-by-step help
- 🧑‍💼 Business Executive → Short, business-focused summary

### 5. Escalation System
Triggers human handoff when:

- Confidence score < threshold
- Sensitive topics detected (billing, refund, legal issues)
- Repeated frustration detected

Outputs structured JSON for human agents.

---

## 🧪 Example Test Cases

| Input | Persona | Behavior |
|------|--------|----------|
| "API returns 401 error" | Technical Expert | Debug steps + logs |
| "It’s not working!! fix this now!" | Frustrated User | Empathy + simple fixes |
| "What’s the ETA for resolution?" | Business Executive | Short business update |
| "I want a refund immediately" | Frustrated User | Escalation to human |

---

## 🛠️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/persona-support-agent.git
cd persona-support-agent
