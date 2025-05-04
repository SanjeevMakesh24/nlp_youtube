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
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked):
        return None

    # Build chunks
    docs, chunk, chunk_start = [], "", None
    for entry in transcript:
        if chunk_start is None:
            chunk_start = entry["start"]
        if len(chunk) + len(entry["text"]) <= 600:
            chunk += " " + entry["text"]
        else:
            docs.append(Document(page_content=chunk.strip(), metadata={"timestamp": chunk_start}))
            chunk, chunk_start = entry["text"], entry["start"]
    if chunk:
        docs.append(Document(page_content=chunk.strip(), metadata={"timestamp": chunk_start}))

    # Embed & search
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    relevant = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join(d.page_content for d in relevant)
    timestamps = [format_ts(d.metadata["timestamp"]) for d in relevant]

    prompt = PromptTemplate.from_template("""
You are an AI assistant helping summarize key points from a YouTube transcript.

Context:
{context}

Question: {question}

Answer:
""")
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question, "link": video_link})

    return {"answer": answer, "timestamps": timestamps}

def summarize_video(video_link):
    video_id = extract_video_id(video_link)
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return "Could not retrieve transcript."

    text = " ".join(e["text"] for e in raw)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                             separators=["\n\n", "\n", ". ", " ", ""])
    docs = splitter.create_documents([text])

    map_prompt = PromptTemplate(template="""
For the excerpt below, write exactly one concise paragraph (3–5 sentences) that captures its main idea:

{text}
""", input_variables=["text"])
    combine_prompt = PromptTemplate(template="""
You are an AI assistant creating a final summary of a YouTube video transcript.
Produce **exactly 2–3 paragraphs** separated by a blank line.
Here are the individual paragraph summaries:
{text}

**Final Summary (2–3 paragraphs):**
""", input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="map_reduce",
                                map_prompt=map_prompt, combine_prompt=combine_prompt)
    result = chain.invoke({"input_documents": docs})
    return result.get("output_text", result)

def main():
    st.set_page_config(page_title="YouTube Chatbot", page_icon="🎬", layout="wide")

    # Theme‐aware bubble + container CSS
    st.markdown("""
    <style>
      .userBubble, .assistantBubble {
        padding:8px; border-radius:5px; margin:4px 0;
      }
      @media (prefers-color-scheme: light) {
        .userBubble      { background:#e6f7ff; color:#000; }
        .assistantBubble { background:#e2e2e2; color:#000; }
      }
      @media (prefers-color-scheme: dark) {
        .userBubble      { background:#4a4a4a; color:#eee; }
        .assistantBubble { background:#333; color:#ddd; }
      }
      .chatContainer {
        border:1px solid #888;
        border-radius:8px;
        padding:8px;
        height:400px;       /* fixed height */
        overflow-y:auto;    /* internal scroll */
        margin-top:8px;
      }
    </style>
    """, unsafe_allow_html=True)

    # Session‐state defaults
    for key, val in {
        "chat_history": [],
        "current_video_id": None,
        "video_title": None,
        "form_submitted": False,
        "last_response": "",
        "new_response_ready": False,
        "previous_question": ""
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.markdown("""
      <div style='text-align:center; margin-bottom:20px;'>
        <h1>🎬 YouTube Chatbot</h1>
        <p>Ask questions about YouTube videos without watching the entire content</p>
      </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # ─── Left: Video Loader & Summary ────────────────────────────────────────────
    with col1:
        st.markdown("<h3>📺 Video Player</h3>", unsafe_allow_html=True)
        url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("📥 Load Video"):
            vid = extract_video_id(url)
            if not vid:
                st.error("Invalid YouTube URL.")
            else:
                st.session_state.chat_history.clear()
                st.session_state.current_video_id = vid
                st.session_state.video_title = url
                st.session_state.chat_history.append({"role":"system",
                                                     "content":f"Video loaded: {url}"})

        if st.session_state.current_video_id:
            st.markdown(
                f"<iframe width='100%' height='315' src='https://www.youtube.com/embed/{st.session_state.current_video_id}' "
                "frameborder='0' allow='autoplay; encrypted-media' allowfullscreen></iframe>",
                unsafe_allow_html=True
            )
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary..."):
                    summ = summarize_video(st.session_state.video_title)
                st.session_state.chat_history.append({"role":"assistant","content":summ})

    # ─── Right: Chat Interface ──────────────────────────────────────────────────
    with col2:
        st.markdown("<h3>💬 Chat</h3>", unsafe_allow_html=True)
        q = st.text_input("Ask a question about the video:", key="user_question")

        # Buttons
        submit, clear = st.columns([1,1])
        with submit:
            ask = st.button("🔍 Submit Question")
        with clear:
            clr = st.button("🗑️ Clear Chat")

        # Clear handler
        if clr:
            st.session_state.chat_history.clear()
            if st.session_state.current_video_id:
                st.session_state.chat_history.append({
                    "role":"system",
                    "content":f"Video loaded: {st.session_state.video_title}"
                })
            st.session_state.form_submitted = False

        # Submit handler
        if ask and q and st.session_state.current_video_id and not st.session_state.form_submitted:
            st.session_state.form_submitted = True
            st.session_state.chat_history.append({"role":"user","content":q})
            with st.spinner("Analyzing video content..."):
                res = query_video(st.session_state.video_title, q) or {}
            if "answer" in res:
                ts = ", ".join(f"[{t}]" for t in res["timestamps"])
                st.session_state.last_response = f"{res['answer']}\n\n**Relevant timestamps:** {ts}"
            else:
                st.session_state.last_response = "Sorry, could not process your question."
            st.session_state.new_response_ready = True

        # Append deferred assistant response
        # Append deferred assistant response
        if st.session_state.new_response_ready and st.session_state.last_response:
            st.session_state.chat_history.append({
                "role":"assistant",
                "content": st.session_state.last_response
            })
            # Reset flags so you can submit again
            st.session_state.new_response_ready = False
            st.session_state.last_response = ""
            st.session_state.form_submitted = False

        # Render chat inside fixed-height container
        # — build one HTML blob for the whole chat
        # add a heading before the container
        chat_html = "<div class='chatContainer'>\n"
        chat_html += "<div style='margin-bottom:8px; font-weight:bold;'>Conversation:</div>\n"
        

        for msg in st.session_state.chat_history:
            if msg["role"] == "system":
                chat_html += f"<div style='text-align:center; color:#888; font-style:italic; margin:4px 0;'>{msg['content']}</div>\n"
            elif msg["role"] == "user":
                chat_html += (
                f"<div class='userBubble'><b>You:</b> {msg['content']}</div>\n"
                )
            else:
                chat_html += (
                f"<div class='assistantBubble'><b>Assistant:</b> {msg['content']}</div>\n"
                )
        chat_html += "</div>"

        # — render it all at once
        st.markdown(chat_html, unsafe_allow_html=True)


        if not st.session_state.current_video_id:
            st.info("👆 Enter a YouTube URL and click 'Load Video' to start chatting.")

if __name__ == "__main__":
    main()
