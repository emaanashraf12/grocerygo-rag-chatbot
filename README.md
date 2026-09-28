# 🛒 GroceryGo – AI-Powered Grocery Delivery Customer Support Chatbot

GroceryGo is a **Retrieval-Augmented Generation (RAG) customer support chatbot** designed for a fictional online grocery delivery service.

The application combines **semantic search, vector retrieval, and a Large Language Model (LLM)** to answer customer questions about orders, delivery, payments, refunds, accounts, and customer support.

Instead of relying only on keyword matching, GroceryGo uses **Sentence Transformer embeddings and FAISS** to retrieve semantically relevant customer-support examples and provides the retrieved information to **Google Gemini** for grounded response generation.

---

## ✨ Features

- 💬 Interactive customer-support chatbot
- 🔎 Semantic search instead of exact keyword matching
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 Kaggle customer-support dataset
- 🗂️ 17,904 processed customer-support records
- 🎯 22 selected customer-support intents
- 📦 6 GroceryGo support categories
- 🔢 384-dimensional sentence embeddings
- ⚡ FAISS vector similarity search
- 🤖 Google Gemini response generation
- 🛡️ Grounded responses to reduce unsupported answers
- 🚫 Fallback handling for out-of-domain questions
- 📖 View retrieved support examples in the UI
- 🧪 Dataset, retrieval, and end-to-end chatbot tests
- 🌐 Streamlit web interface

---

## 🧠 How It Works

GroceryGo follows a Retrieval-Augmented Generation pipeline:

```text
Kaggle Bitext Dataset
        ↓
Data Cleaning & Intent Filtering
        ↓
17,904 Processed Customer Utterances
        ↓
LangChain Documents
        ↓
all-MiniLM-L6-v2 Embeddings
        ↓
FAISS Vector Store
        ↓
User Question
        ↓
Question Embedding
        ↓
Semantic Similarity Search
        ↓
Top Relevant Support Examples
        ↓
Dominant Intent Selection
        ↓
Approved GroceryGo Response Context
        ↓
Google Gemini
        ↓
Final Customer Response
```

When a user submits a question, the system retrieves the most semantically similar customer-support examples rather than searching for exact keyword matches.

The chatbot initially retrieves the **top 5 similar documents**. It then determines the dominant intent among the retrieved documents and selects up to **3 documents from that intent** as context for Gemini.

---

## 🗃️ Dataset

The project uses the **Bitext customer-support dataset** obtained from Kaggle.

### Original Dataset

| Property | Value |
|---|---:|
| Records | 21,534 |
| Features | 4 |
| Categories | 11 |
| Intents | 27 |

Original features:

- `flags`
- `utterance`
- `category`
- `intent`

### Processed GroceryGo Dataset

After cleaning, duplicate removal, scope filtering, and intent selection:

| Property | Value |
|---|---:|
| Processed Records | 17,904 |
| Selected Intents | 22 |
| Display Categories | 6 |
| LangChain Documents | 17,904 |

### GroceryGo Categories

| Category | Records |
|---|---:|
| Payment | 4,635 |
| Account | 4,557 |
| Support | 3,827 |
| Orders | 2,249 |
| Refunds | 1,929 |
| Delivery | 707 |
| **Total** | **17,904** |

The original Bitext dataset contains customer utterances and intent labels rather than GroceryGo-specific answers. Therefore, the selected intents are mapped to **controlled GroceryGo responses** created for this project.

---

## 🎯 Supported Intents

The final knowledge base contains 22 customer-support intents:

```text
create_account
recover_password
registration_problems
edit_account
delete_account
switch_account

place_order
change_order
cancel_order
track_order

delivery_options
delivery_period
change_shipping_address
set_up_shipping_address

check_payment_methods
payment_issue

get_refund
check_refund_policy
track_refund

contact_customer_service
contact_human_agent
complaint
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and preprocessing |
| LangChain | RAG orchestration and document handling |
| Sentence Transformers | Semantic text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| Google Gemini | Grounded response generation |
| Streamlit | Interactive web interface |
| python-dotenv | Environment variable management |

---

## 🔎 Semantic Embeddings

GroceryGo uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts customer questions into **384-dimensional dense vectors** representing their semantic meaning.

This allows questions such as:

```text
Where is my order? I want to track it.
```

to match dataset examples such as:

```text
I wanna track my order, can you tell me where I can do it?
```

even though the sentences are not identical.

---

## 🧪 Retrieval Evaluation

A small evaluation set containing six paraphrased customer questions was used to test semantic retrieval.

| Query Type | Expected Intent | Result |
|---|---|---|
| Password recovery | `recover_password` | ✅ Correct |
| Order tracking | `track_order` | ✅ Correct |
| Payment methods | `check_payment_methods` | ✅ Correct |
| Refund request | `get_refund` | ✅ Correct |
| Delivery period | `delivery_period` | ✅ Correct |
| Human support | `contact_human_agent` | ✅ Correct |

### Result

```text
6/6 test queries retrieved the expected intent at rank 1.
```

> **Note:** This result refers only to the six-query evaluation set and should not be interpreted as 100% accuracy across all possible customer questions.

---

## 💬 Example Conversations

### Order Tracking

**User:**

```text
Where is my order? Can I track it?
```

**GroceryGo:**

```text
You can check the current status of your order from the
order tracking or order details section.
```

---

### Delivery

**User:**

```text
How long does delivery take?
```

**GroceryGo:**

```text
The available delivery period or estimated delivery time
is shown when arranging delivery for your order.
```

---

### Refund

**User:**

```text
I want to request a refund.
```

**GroceryGo:**

```text
You can request a refund through the appropriate order-help
or customer-support option. Refund approval depends on the
applicable refund policy and the details of the request.
```

---

### Out-of-Domain Question

**User:**

```text
Who invented the telephone?
```

**GroceryGo:**

```text
I don't have enough information in the GroceryGo knowledge
base to answer that question. Please contact customer support.
```

The fallback behavior helps prevent the chatbot from answering unrelated questions using the LLM's general knowledge.

---

## 📁 Project Structure

```text
grocery_faq_chatbot/
│
├── app.py
├── bitext_customer_support.csv
├── requirements.txt
├── .gitignore
├── test_data.py
├── test_vector.py
├── test_chatbot.py
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── vector_store.py
    └── chatbot.py
```

> `.env` and `.venv` are intentionally excluded from the repository.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL above with the URL of this repository.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Google Gemini API Configuration

The application requires a Google Gemini API key.

Create a file named:

```text
.env
```

in the project root.

Add:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### Important

Never commit your real API key to GitHub.

The `.gitignore` file should contain:

```gitignore
.env
.venv/
__pycache__/
*.pyc
faiss_index/
```

---

## ▶️ Run the Application

Start GroceryGo with:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL that can be opened in a browser.

---

## 🧪 Run the Tests

### Dataset Preprocessing Test

```bash
python test_data.py
```

This verifies dataset loading and preprocessing.

Expected processed dataset size:

```text
Processed records: 17904
```

### Semantic Retrieval Test

```bash
python test_vector.py
```

This evaluates whether FAISS retrieves the expected intent for the test queries.

The current six-query evaluation produced:

```text
6/6 test queries retrieved the expected intent at rank 1.
```

### End-to-End RAG Test

```bash
python test_chatbot.py
```

This tests the complete pipeline:

```text
Question
   ↓
Embedding
   ↓
FAISS Retrieval
   ↓
Intent Selection
   ↓
Approved Context
   ↓
Gemini
   ↓
Answer
```

---

## 🛡️ Grounding and Hallucination Control

The Gemini system prompt instructs the model to answer using only the approved information supplied by the GroceryGo knowledge base.

The model is instructed not to invent:

- prices
- discounts
- delivery times
- payment methods
- refund guarantees
- company policies
- unsupported procedures

If sufficient information is unavailable, the chatbot uses a controlled fallback response.

---

## ⚠️ Limitations

This project is a demonstration system and not a production grocery-delivery platform.

Current limitations include:

- The retrieval evaluation uses a small six-query test set.
- FAISS returns nearest neighbors even for unrelated questions.
- There is currently no calibrated similarity threshold for out-of-domain detection.
- GroceryGo responses are project-defined rather than real company policies.
- The chatbot does not connect to live orders or customer accounts.
- It does not process real payments or refunds.
- It does not access real inventory or delivery-driver information.
- The application currently focuses on English.
- Gemini API quotas and network availability can affect response generation.

---

## 🔮 Future Improvements

Future versions could include:

- Larger held-out evaluation dataset
- Top-1 and Top-k retrieval metrics
- Precision, recall, and F1 evaluation
- Mean Reciprocal Rank (MRR)
- Similarity threshold for out-of-domain detection
- Dedicated intent classifier
- Retrieval reranking
- Persistent FAISS index
- Multilingual embeddings
- Multilingual customer support
- Real order-tracking API integration
- Inventory integration
- Payment and refund integrations
- Human-agent escalation
- Authentication and authorization
- Improved privacy and security controls

---

## 🎓 Project Purpose

GroceryGo was developed as an educational AI/NLP project to demonstrate the practical application of:

- Natural Language Processing
- Transformer-based sentence embeddings
- Semantic search
- Vector databases
- Retrieval-Augmented Generation
- Prompt engineering
- Large Language Models
- Customer-support chatbot design

---

## 📚 Key Resources

- Bitext Customer Support Dataset – Kaggle
- Sentence Transformers – Hugging Face
- FAISS – Meta AI Research
- LangChain
- Google Gemini API
- Streamlit

---

## 👤 Author

**Emaan Ashraf**

AI / Machine Learning Project

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐.

Contributions, suggestions, and feedback are welcome.
