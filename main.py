from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain


#environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()  #load OPENAI_API_KEY

#flask backend
app = Flask(__name__)
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
@app.route('/query', methods=['POST'])
def query_video():
    data = request.get_json(force=True)
    video_link = data.get('video_link')
    question = data.get('question')
    if not video_link or not question:
        return jsonify({'error': 'Missing `video_link` or `question` field'}), 400

    video_id = extract_video_id(video_link)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked) as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500

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

    return jsonify({'answer': response, 'timestamps': top_timestamps})


# Add new route for summarization
@app.route('/summarize', methods=['POST'])
def summarize_video():
    data = request.get_json(force=True)
    video_link = data.get('video_link')

    if not video_link:
        return jsonify({'error': 'Missing video_link'}), 400

    try:
        video_id = extract_video_id(video_link)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, RequestBlocked) as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500

    # Concatenate entire transcript
    full_text = " ".join([entry['text'] for entry in transcript])
    start_time = transcript[0]['start'] if transcript else 0

    # Optionally chunk if it's too long for your LLM's context window
    if len(full_text) > 3000:  # adjust this limit based on your LLM context
        chunk_size = 1000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        summaries = []
        for chunk in chunks:
            prompt = PromptTemplate.from_template("""
            Summarize the following transcript section clearly and extremely concisely (1 or 2 sentences):

            {context}

            Summary:
            """)
            chain = prompt | llm
            summary = chain.invoke({'context': chunk})
            summaries.append(summary)
        final_summary = "\n".join(summaries)
    else:
        prompt = PromptTemplate.from_template("""
        Summarize the following YouTube transcript clearly and concisely:

        {context}

        Summary:
        """)
        chain = prompt | llm
        final_summary = chain.invoke({'context': full_text})

    return jsonify({'summary': final_summary, 'video_link': video_link})

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>YouTube RAG API</h1>
    """, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

#scope for improvements:
#longer videos --> more chunks to the llm