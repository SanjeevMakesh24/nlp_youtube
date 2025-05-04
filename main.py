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

def summarize_video(video_link):
    print(f"📺 Processing video: {video_link}")
    
    #extract video ID from the link
    video_id = extract_video_id(video_link)
    print(f"🔑 Extracted video ID: {video_id}")
    
    try:
        #get transcript
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"✓ Successfully retrieved transcript with {len(raw)} entries")
        
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"❌ Transcript Error: {e}")
        return f"Could not retrieve transcript: {str(e)}"
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return f"An unexpected error occurred: {str(e)}"

    #join all transcript entries into a single text
    text = " ".join(entry["text"] for entry in raw)
    print(f"📝 Total transcript length: {len(text)} characters")
    
    # Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  #each chunk will be about 1000 characters
        chunk_overlap=200,  #200 character overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    docs = splitter.create_documents([text])
    print(f"🔖 Split transcript into {len(docs)} chunks")

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


#run directly
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Ask a question about the video")
    print("2. Summarize the video")
    choice = input("Enter 1 or 2: ").strip()

    video_link = input("Enter YouTube video link: ").strip()

    if choice == "1":
        question = input("Enter your question: ").strip()
        query_video(video_link, question)
    elif choice == "2":
        summarize_video(video_link)
    else:
        print("Invalid choice. Please enter 1 or 2.")


#scope for improvements:
#longer videos --> more chunks to the llm

#Summarize the following transcript section clearly and extremely concisely (1 or 2 sentences):