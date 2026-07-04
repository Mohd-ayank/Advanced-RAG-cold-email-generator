# Advanced-RAG-cold-email-generator : AI-Powered Job Outreach

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-green)
![Weaviate](https://img.shields.io/badge/Weaviate-Vector_DB-orange)
![Cohere](https://img.shields.io/badge/Cohere-Rerank-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

An enterprise-grade, end-to-end AI application that generates highly personalized cold emails for software engineering job applications. 

By leveraging an **Advanced Retrieval-Augmented Generation (RAG)** architecture—combining **Hybrid Search (Vector + BM25)** and **Cohere Reranking**—this tool perfectly aligns a company's job description with the most relevant open-source projects from a GitHub portfolio.

---

## Application Showcase

> **Note to self:** *Upload your UI screenshots to a `docs/` or `assets/` folder in your repo and replace the links below!*

![App UI Screenshot]()
*Figure 1: The Streamlit UI where a user inputs a job URL and views the extracted skills alongside the final generated email.*

---

## Why is this project "Brilliant"? (The Architecture)

Standard RAG pipelines often fail in technical domains because purely semantic vector searches miss crucial exact-keyword matches (e.g., matching "React Native" specifically, not just "Mobile Frameworks"). Furthermore, feeding too many retrieved documents to an LLM causes "Lost in the Middle" syndrome, degrading output quality.

**This project solves these issues with a production-ready architecture:**

1. **Automated ETL Pipeline:** A custom script pulls real repositories from the GitHub API, cleans the Markdown/HTML noise, and uses an LLM (Groq/Llama-3.3) to extract structured JSON metadata (Skills, Concepts, Architecture) before chunking.
2. **Hybrid Search:** Uses **Weaviate** to perform an `alpha=0.5` hybrid search. This perfectly balances **Dense Vector Search** (for conceptual matching) with **Sparse BM25 Search** (for exact tech-stack keyword matching).
3. **Cross-Encoder Reranking:** Takes the broad top 10 results from the Hybrid Search and passes them through **Cohere's Rerank-v3.5 model** to contextually compress the results down to the absolute best 2-3 projects.
4. **Contextual Email Generation:** Feeds the highly-curated portfolio context and the scraped job description into a high-speed LLM to draft a persuasive, accurate, and preamble-free cold email.

---

## Project Structure

The codebase strictly adheres to software engineering best practices, cleanly separating the **Offline Data Ingestion (ETL)** from the **Online Application**.

```text
cold-email-gen/
│
├── data/                            # Local storage for CSV datasets
│
├── scripts/                         # OFFLINE ETL PIPELINE
│   ├── 1_fetch_github_data.py       # Scrapes top tech repos via GitHub API
│   ├── 2_clean_and_extract.py       # Cleans data & uses LLM to parse structured JSON
│   └── 3_ingest_db.py               # Chunks text, generates embeddings, uploads to Weaviate
│
├── src/                             # ONLINE APPLICATION
│   ├── config.py                    # Environment variable management
│   ├── scraper.py                   # LangChain WebBaseLoader for Job URLs
│   ├── retriever.py                 # Weaviate Hybrid Search & Cohere Reranker logic
│   └── llm_service.py               # Groq LLM logic for parsing JDs and writing emails
│
├── app.py                           # Streamlit Web UI
├── requirements.txt                 # Project dependencies
└── .env                             # Secret API Keys (Git-ignored)
```
---

## ⚙️ Tech Stack

* **UI Framework:** Streamlit
* **Orchestration:** LangChain
* **LLM Provider:** Groq (`llama-3.3-70b-versatile`) for lightning-fast inference
* **Vector Database:** Weaviate Cloud
* **Embeddings:** HuggingFace (`BAAI/bge-base-en-v1.5`) runs locally for free/fast embedding
* **Reranker:** Cohere (`rerank-v3.5`)
* **Data Processing:** Pandas, Regex, GitHub REST API

---

## 🚀 Getting Started

### 1. Prerequisites
You will need API keys for:
* [Groq](https://console.groq.com/) (Free)
* [Weaviate Cloud](https://console.weaviate.cloud/) (Free Sandbox)
* [Cohere](https://cohere.com/) (Free Trial Key)
* [HuggingFace](https://huggingface.co/) (Free Access Token)
* [GitHub](https://github.com/settings/tokens) (Classic Personal Access Token)

### 2. Installation
Clone the repository and set up your virtual environment:
```bash
git clone https://github.com/YOUR_USERNAME/advanced-rag-cold-email-gen.git
cd advanced-rag-cold-email-gen

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your keys:
```env
GITHUB_TOKEN="your_github_token"
GROQ_API_KEY="your_groq_key"
WEAVIATE_URL="your_cluster.aws.weaviate.cloud"
WEAVIATE_API_KEY="your_weaviate_api_key"
HF_TOKEN="your_huggingface_token"
COHERE_API_KEY="your_cohere_api_key"
```

---

## 🛠️ Usage

### Phase 1: Populate the Database (ETL)
Run these scripts **once** to build your Vector Database.
```bash
# 1. Download repository data from GitHub
python scripts/1_fetch_github_data.py

# 2. Clean text and extract structured JSON via LLM
python scripts/2_clean_and_extract.py

# 3. Chunk, embed, and ingest into Weaviate
python scripts/3_ingest_db.py
```

### Phase 2: Run the App
Once your database is populated, start the Streamlit UI to generate emails!
```bash
streamlit run app.py
```
1. Paste a Job Posting URL (e.g., from the Nike careers page).
2. Watch the progress tracker execute the RAG pipeline.
3. Review the extracted skills, matched portfolio projects, and the final generated email.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Mohd-ayank/advanced-rag-cold-email-gen/issues).
