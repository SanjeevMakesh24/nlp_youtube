from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

llm = OpenAI(temperature=0.2)

def extract_video_id(url):
    return url.split("v=")[-1].split("&")[0]

def format_ts(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def query_video(video_link, question):
    video_id = extract_video_id(video_link)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked, Exception) as e:
        print(f"Error: {e}")
        return

    docs = []
    chunk = ""
    chunk_start = None
    max_chars = 600
    for entry in transcript:
        if chunk_start is None:
            chunk_start = entry['start']
        if len(chunk) + len(entry['text']) <= max_chars:
            chunk += " " + entry['text']
        else:
            docs.append(Document(page_content=chunk.strip(), metadata={'timestamp': chunk_start}))
            chunk = entry['text']
            chunk_start = entry['start']
    if chunk:
        docs.append(Document(page_content=chunk.strip(), metadata={'timestamp': chunk_start}))

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    relevant_docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    top_timestamps = [format_ts(doc.metadata.get('timestamp', 0)) for doc in relevant_docs]

    prompt = PromptTemplate.from_template("""
    You are an AI assistant helping summarize key points from a YouTube transcript.

    Use the context below to answer the question thoroughly and accurately, reflecting what was actually said in the video.

    Context:
    {context}

    Question: {question}

    INSTRUCTIONS:
    1. Focus exclusively on information mentioned in the transcript
    2. Provide direct quotes when possible to support key points
    3. Maintain the original meaning and nuance from the video
    4. If the transcript doesn't contain relevant information to answer the question, clearly state this
    5. Organize your response with clear headings if appropriate
    6. If you need more context, refer to the video link: {link}

    Answer:
    """)
    chain = prompt | llm
    response = chain.invoke({'context': context, 'question': question, 'link': video_link})

    return {
        "answer": response,
        "timestamps": top_timestamps
    }

def summarize_video(video_link):
    video_id = extract_video_id(video_link)
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception) as e:
        return f"Could not retrieve transcript: {e}"

    text = " ".join(entry["text"] for entry in raw)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.create_documents([text])

    map_prompt = PromptTemplate(template="""
    For the excerpt below, write exactly one concise paragraph (3–5 sentences) that captures its main idea:

    {text}
    """, input_variables=["text"])
    combine_prompt = PromptTemplate(template="""
    You are an AI assistant creating a final summary of a YouTube video transcript.
    Produce **exactly 2–3 paragraphs** separated by a blank line.
    Each paragraph should be no more than 5-6 sentences.
    Highlight the key themes and include direct quotes sparingly.
    If the transcript doesn't cover something, omit it—do not speculate.
    But don't finish a sentence halfway; just finish it no matter what.

    Here are the individual paragraph summaries:
    {text}

    **Final Summary (2–3 paragraphs):**
    """, input_variables=["text"])

    summarize_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False
    )
    result = summarize_chain.invoke({"input_documents": docs})
    return result.get("output_text", result)

def main():
    st.set_page_config(page_title="YouTube Chatbot", page_icon="🎬", layout="wide")

    # theme‑aware styles
    st.markdown("""
    <style>
    .userBubble, .assistantBubble {
      padding:8px;
      border-radius:5px;
      margin:4px 0;
    }
    /* Light mode */
    @media (prefers-color-scheme: light) {
      .userBubble      { background-color: #e6f7ff; color: #000; }
      .assistantBubble { background-color: #e2e2e2; color: #000; }
    }
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
      .userBubble      { background-color: #4a4a4a; color: #eee; }
      .assistantBubble { background-color: #333333; color: #ddd; }
    }
    /* Chat container styling */
    .chatContainer {
      border:1px solid #888;
      border-radius:8px;
      padding:8px;
      max-height:400px;
      overflow-y:auto;
      margin-top:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    defaults = {
        'chat_history': [],
        'current_video_id': None,
        'video_title': None,
        'summary_generated': False,
        'form_submitted': False,
        'last_response': "",
        'new_response_ready': False,
        'previous_question': ""
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1>🎬 YouTube Chatbot</h1>
        <p>Ask questions about YouTube videos without watching the entire content</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # Left column: video loader + summary
    with col1:
        st.markdown("<h3>📺 Video Player</h3>", unsafe_allow_html=True)
        video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("📥 Load Video"):
            vid = extract_video_id(video_url)
            if not vid:
                st.error("Invalid YouTube URL. Please check and try again.")
            else:
                st.session_state.chat_history = []
                st.session_state.summary_generated = False
                st.session_state.current_video_id = vid
                st.session_state.video_title = video_url
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"Video loaded: {video_url}"
                })

        if st.session_state.current_video_id:
            st.markdown(
                f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{st.session_state.current_video_id}" '
                'frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_video(st.session_state.video_title)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": summary
                    })
                    st.session_state.summary_generated = True

    # Right column: chat interface
    with col2:
        st.markdown("<h3>💬 Chat</h3>", unsafe_allow_html=True)

        user_question = st.text_input("Ask a question about the video:", key="user_question")
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_button = st.button("🔍 Submit Question", key="submit")
            if user_question != st.session_state.previous_question:
                st.session_state.form_submitted = False
                st.session_state.previous_question = user_question
        with col_clear:
            clear_chat = st.button("🗑️ Clear Chat")

        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.form_submitted = False
            if st.session_state.current_video_id:
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"Video loaded: {st.session_state.video_title}"
                })

        if submit_button and user_question and st.session_state.current_video_id and not st.session_state.form_submitted:
            st.session_state.form_submitted = True
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            with st.spinner("Analyzing video content..."):
                result = query_video(st.session_state.video_title, user_question)
            if result and "answer" in result:
                ts = ", ".join(f"[{t}]" for t in result["timestamps"])
                st.session_state.last_response = f"{result['answer']}\n\n**Relevant timestamps:** {ts}"
            else:
                st.session_state.last_response = "Sorry, I couldn't process your question. Please try again."
            st.session_state.new_response_ready = True

        if st.session_state.new_response_ready and st.session_state.last_response:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": st.session_state.last_response
            })
            st.session_state.new_response_ready = False
            st.session_state.last_response = ""

        st.markdown("<div class='chatContainer'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "system":
                st.markdown(
                    f"<div style='text-align:center; color:#888; font-style:italic; margin:4px 0;'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "user":
                st.markdown(
                    f"<div class='userBubble'><b>You:</b> {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='assistantBubble'><b>Assistant:</b> {msg['content']}</div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Instructions when no video loaded
        if not st.session_state.current_video_id:
            st.info("👆 Enter a YouTube URL and click 'Load Video' to start chatting.")

if __name__ == "__main__":
    main()
