# YouTube ChatBot
## Ask questions and get instant answer about a video

## 🚀 Features & Architecture
- **Transcript Extraction**  
  Fetches YouTube captions (automatic or community-generated) via `youtube_transcript_api`, with optional rotating-proxy support to avoid IP blocks.  
- **Chunking & Embedding**  
  Uses LangChain’s `RecursiveCharacterTextSplitter` to break long transcripts into ~6000-char chunks (100-char overlap), then embeds each chunk with a HuggingFace Sentence-Transformer (`all-MiniLM-L6-v2`).  
- **FAISS–Powered Retrieval**  
  Loads embeddings into a FAISS index for fast Nearest-Neighbor search. Queries return only the top-k relevant chunks.  
- **LLM Q&A**  
  Builds a prompt using those top-k chunks and your question, then calls OpenAI Chat GPT via LangChain’s LLM chain—ensuring every answer is from within the video.  
- **Two-Stage Map-Reduce Summarization**  
  1. **Map step**: Each chunk → one concise paragraph.  
  2. **Reduce step**: Stitch those mini-summaries into exactly 2–3 well-structured paragraphs with clear instructions to avoid hallucination.  
- **Streamlit Front-End**  
  Interactive, two-panel UI:  
  - Left: Load video.  
  - Right: Ask questions and follow-ups in a chat container, with clickable timestamps.

## Setup Instructions

1. Close the Repository:
	git clone https://github.com/SanjeevMakesh24/nlp_youtube.git
	cd nlp_youtube
2. Create a virtual Environment:
	python3 -m venv .venv
3. Activate the virtual Environment:
	source .venv/bin/activate
4. Install dependencies:
	pip install -r requirements.txt
5. Create a .env file in the project root:
   	OPENAI_API_KEY=sk-YOUR_KEY_HERE
6. Run the project:
	streamlit run nlp.py
