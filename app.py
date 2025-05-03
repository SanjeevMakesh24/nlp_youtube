import streamlit as st
import requests

st.set_page_config(page_title=" YouTube Transcript QA", page_icon="🎥", layout="centered")

# --- Page Title ---
st.title("🎬 YouTube Transcript Question-Answering Bot")

st.markdown(
    """
    Welcome to the **YouTube RAG Chatbot**! 🎯  
    Ask any question about a YouTube video without watching the full video.
    
    ---
    """,
    unsafe_allow_html=True
)

# --- User Inputs ---
st.header(" Provide Video Details")
video_url = st.text_input("🔗 YouTube Video URL")
user_query = st.text_input("❓ Your Question about the Video")

# --- Query Button ---
if st.button("🔎 Get Answer"):
    if not video_url or not user_query:
        st.warning("⚠️ Please enter both the video URL and your question!")
    else:
        with st.spinner("🔄 Fetching and analyzing... Please wait!"):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={
                        "video_link": video_url,
                        "question": user_query
                    }
                )

                if response.status_code == 200:
                    result = response.json()

                    # --- Display Answer ---
                    st.success("✅ Got the answer!")
                    st.markdown("###  **Answer:**")
                    st.write(result['answer'])

                    # --- Display Timestamps ---
                    st.markdown("### 🕒 **Relevant Timestamps:**")
                    for idx, ts in enumerate(result['timestamps']):
                        st.markdown(f"**{idx+1}.** `[{ts}]`")

                    # --- Show YouTube logo ---
                    st.markdown("---")
                    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=200)

                else:
                    error_message = response.json().get("error", "Unknown error occurred.")
                    st.error(f"❌ Error: {error_message}")

            except Exception as e:
                st.error(f"🚨 Request failed: {e}")

