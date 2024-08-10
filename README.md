# AI Powered Economics Tutor

AI Powered Economics Tutor is an innovative bilingual chatbot designed to address the challenges faced by students and enthusiasts in keeping up with the rapidly evolving field of economics. It provides real-time updates and detailed explanations in both English and Urdu, enhancing the learning experience with up-to-date and comprehensible information.

## Live Demo

Check out the live demo of AI Powered Economics Tutor [here](https://huggingface.co/spaces/rayyanphysicist/EconomyChatBot).

## Features

- **PDF Processing:** The Cambridge book is read and converted into manageable chunks for efficient processing.
  
- **Embedding Creation:** Chunks are transformed into embeddings using Google embeddings, enabling effective information retrieval.
  
- **Vector Storage:** Embeddings are stored in a FAISS vector database for quick and efficient access.

- **Retrieval-Augmented Generation (RAG):** Combines the power of retrieval and generation to provide accurate and contextually relevant responses. This feature allows the chatbot to pull information from stored embeddings and generate refined answers tailored to the user's query.

- **Chatbot Development:**
    - **Initial Response Generation:** Gemini-pro LLM generates initial responses to user queries.
    - **Response Refinement and Translation:** FALCON 180B refines these responses and translates them into Urdu.

- **User Interaction:** All user queries are saved as history for future reference, allowing users to revisit past interactions. 

## Installation

To install AI Powered Economics Tutor, follow these steps:

1. Clone the repository:

```
git clone https://github.com/mrnust/SmartEconomicStudyCompanion_Multi-AgentPdfQ-ACHATBOT.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the load.py script to upload PDFs to the database:

```
python load.py
```

4. Run the application:

```
python app.py
```

