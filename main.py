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

#environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()  #load OPENAI_API_KEY

llm = OpenAI(temperature=0.2)

#to extract YouTube video ID
def extract_video_id(url):
    return url.split("v=")[-1].split("&")[0]

#to format the seconds into minute:second
def format_ts(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{minutes}:{secs:02d}"

#to ask specific questions about the video
def query_video(video_link, question):
    video_id = extract_video_id(video_link)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked) as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    #split the long transcript into smaller chunks
    #chunk_overlap: last 100 characters of one chunk will appear at the beginning of the next chunk

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

    #similarity search
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embedding_model)
    relevant_docs = vectorstore.similarity_search(question, k=3) #get top 3 chunks

    #extract timestamps and context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    top_timestamps = [format_ts(doc.metadata.get('timestamp', 0)) for doc in relevant_docs]

    #prompt template for llm
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
    """
    )

    chain = prompt | llm
    response = chain.invoke({'context': context, 'question': question, 'link': video_link})

    print("\n--- Answer ---\n")
    print(response)
    print("\nTimestamps of relevant parts:", top_timestamps)

    return {
        "answer": response,          
        "timestamps": top_timestamps 
    }

def summarize_video(video_link):
    print(f"Processing video: {video_link}")
    
    #extract video ID from the link
    video_id = extract_video_id(video_link)
    print(f"Extracted video ID: {video_id}")
    
    try:
        #get transcript
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"✓ Successfully retrieved transcript with {len(raw)} entries")
        
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"Transcript Error: {e}")
        return f"Could not retrieve transcript: {str(e)}"
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"An unexpected error occurred: {str(e)}"

    #join all transcript entries into a single text
    text = " ".join(entry["text"] for entry in raw)
    print(f"Total transcript length: {len(text)} characters")
    
    # Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  #each chunk will be about 1000 characters
        chunk_overlap=200,  #200 character overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    docs = splitter.create_documents([text])
    print(f"split transcript into {len(docs)} chunks")

    #process each chunk into a concise paragraph
    map_prompt = PromptTemplate(
        template="""
    For the excerpt below, write exactly one concise paragraph (3–5 sentences) that captures its main idea:

    {text}
    """,
        input_variables=["text"],
    )

    #merge chunked summaries into a final 2-3 paragraph summary
    combine_prompt = PromptTemplate(
        template="""
    You are an AI assistant creating a final summary of a YouTube video transcript.
    Produce **exactly 2–3 paragraphs** separated by a blank line.
    Each paragraph should be no more than 5-6 sentences.
    Highlight the key themes and include direct quotes sparingly.
    If the transcript doesn't cover something, omit it—do not speculate.
    But don't finish a sentance halfway, just finish it no matter what.

    Here are the individual paragraph summaries:
    {text}

    **Final Summary (2–3 paragraphs):**
    """,
        input_variables=["text"],
    )

    #create and run the map-reduce chain
    summarize_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False  # Set to True for debugging
    )

    result = summarize_chain.invoke({"input_documents": docs})
    final_summary = result.get("output_text", result)
    
    print("\n--- Video Summary ---\n")
    print(final_summary)
    return final_summary


#run directly
# Main Streamlit App
def main():
    st.set_page_config(
        page_title="YouTube Chatbot",
        page_icon="🎬",
        layout="wide"
    )

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None
        
    if 'video_title' not in st.session_state:
        st.session_state.video_title = None
        
    if 'summary_generated' not in st.session_state:
        st.session_state.summary_generated = False
        
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    # App title and header
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1>🎬 YouTube Chatbot</h1>
            <p>Ask questions about YouTube videos without watching the entire content</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3>📺 Video Player</h3>", unsafe_allow_html=True)
        
        # Video URL input
        video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        
        # Process video URL
        if st.button("📥 Load Video"):
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
            else:
                # Reset chat history when loading a new video
                if st.session_state.current_video_id != video_id:
                    st.session_state.chat_history = []
                    st.session_state.summary_generated = False
                
                st.session_state.current_video_id = video_id
                st.session_state.video_title = video_url  # Using URL as title for now
                
                # Add system message to chat history
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"Video loaded: {st.session_state.video_title}"
                })
        
        # Display the YouTube video if ID is available
        if st.session_state.current_video_id:
            st.markdown(
                f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{st.session_state.current_video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )
            
            # Video controls
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_video(video_url)  # Use the full URL
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": summary
                    })
                    st.session_state.summary_generated = True
                
    with col2:
        st.markdown("<h3>💬 Chat</h3>", unsafe_allow_html=True)

        # start a scrollable container at 400px tall
        st.markdown(
            """
            <div style="
                border:1px solid #ddd;
                border-radius:8px;
                padding:8px;
                max-height:400px;
                overflow-y:auto;
                background-color:#f9f9f9;
            ">
            """,
            unsafe_allow_html=True,
        )

        # now dump all of chat_history inside that div
        for message in st.session_state.chat_history:
            if message["role"] == "system":
                st.markdown(f"""
                    <div style='text-align:center; color:#555; font-style:italic; margin:4px 0;'>
                    {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "user":
                st.markdown(f"""
                    <div style='background-color:#e6f7ff; padding:8px; border-radius:5px; margin:4px 0;'>
                    <b>You:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:  # assistant
                st.markdown(f"""
                    <div style='background-color:#f0f0f0; padding:8px; border-radius:5px; margin:4px 0;'>
                    <b>Assistant:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

        # close the scrollable div
        st.markdown("</div>", unsafe_allow_html=True)

        # below that div, your input box and buttons
        user_question = st.text_input("Ask a question about the video:", key="user_question")
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_button = st.button("🔍 Submit Question", key="submit")
            # Reset form_submitted when the question changes
            if 'previous_question' not in st.session_state:
                st.session_state.previous_question = ""
            if user_question != st.session_state.previous_question:
                st.session_state.form_submitted = False
                st.session_state.previous_question = user_question
        
        with col_clear:
            clear_chat = st.button("🗑️ Clear Chat")
            
                    # Process user question
        if submit_button and user_question and st.session_state.current_video_id:
            # Store the question to process
            question_to_process = user_question
            
            # Create a form submit state to prevent duplicate processing
            if 'form_submitted' not in st.session_state:
                st.session_state.form_submitted = False
                
            if not st.session_state.form_submitted:
                st.session_state.form_submitted = True
                
                # Add user question to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question_to_process
                })
                
                # Process the question
                with st.spinner("Analyzing video content..."):
                    video_url = st.session_state.video_title  # Use stored URL
                    result = query_video(video_url, question_to_process)
                    
                    if result and "answer" in result:
                        # Format the response with timestamps
                        timestamps_formatted = ", ".join([f"[{ts}]" for ts in result["timestamps"]])
                        response = f"{result['answer']}\n\n**Relevant timestamps:** {timestamps_formatted}"
                        
                        # Add response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Sorry, I couldn't process your question. Please try again."
                        })
                
               
            else:
                # Reset the form submission state for the next question
                st.session_state.form_submitted = False
                
        # Clear chat history
        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.form_submitted = False
            if st.session_state.current_video_id:
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"Video loaded: {st.session_state.video_title}"
                })
           
        
        # Instructions when no video is loaded
        if not st.session_state.current_video_id:
            st.info("👆 Enter a YouTube URL and click 'Load Video' to start chatting.")

if __name__ == "__main__":
    main()