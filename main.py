from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#to load variables from .env file
load_dotenv()

llm = OpenAI(temperature=0.2)

#function to extract video ID from URL
def extract_video_id(url):

    #https://www.youtube.com/watch?v=042pDj9FJ7Y&ab_channel=TEDxTalks
    #.split("&")[0] to remove any additional parameters
    
    return url.split("v=")[-1].split("&")[0]

#video link
video_link = "https://www.youtube.com/watch?v=Kbk9BiPhm7o&t=1s"  
video_id = extract_video_id(video_link)

try:
    #get transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    #join all text segments
    docs = []
    for entry in transcript:
        docs.append(Document(page_content=entry["text"], metadata={"timestamp": entry["start"]})) #to get the timestamp

    #split the long transcript into smaller chunks
    #chunk_overlap: last 100 characters of one chunk will appear at the beginning of the next chunk

    docs = []
    chunk = ""
    chunk_start = None
    max_chars = 600

    for entry in transcript:
        if chunk_start is None:
            chunk_start = entry["start"]

        if len(chunk) + len(entry["text"]) <= max_chars:
            chunk += " " + entry["text"]
        else:
            docs.append(Document(page_content=chunk.strip(), metadata={"timestamp": chunk_start}))
            chunk = entry["text"]
            chunk_start = entry["start"]

    #add the last chunk
    if chunk:
        docs.append(Document(page_content=chunk.strip(), metadata={"timestamp": chunk_start}))


    print(f"\n Created {len(docs)} chunks \n")

    #a sentence-transformer model for embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #FAISS vector store from the chunks
    vectorstore = FAISS.from_documents(docs, embedding_model)


    question = "What do they talk about neurosurgery?"
    relevent_docs = vectorstore.similarity_search(question, k=3) #list

    #print(type(relevent_docs))

    #top matching chunks
    print("\nTop Matching Transcript Chunks:")
    for i, doc in enumerate(relevent_docs):
        print(f"\n--- Chunk {i+1} ---")
        timestamp = doc.metadata.get("timestamp", "N/A")
        print(f"[Timestamp: {timestamp:.2f} sec]")
        print(doc.page_content)

    
    #prompt template
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
    5. Organize your response with clear structure and headings if appropriate
    6. If you need more context from the video, here is the video link {link}

    Answer:
    """)

    context = "\n\n".join([doc.page_content for doc in relevent_docs])

    chain = prompt | llm

    # Run the LLM to get a final answer
    response = chain.invoke({
        "context": context,
        "question": question,
        "link": video_link
    })

    print("\n Final Answer:")
    print(response)
    

except Exception as e:
    print(f"Error: {e}")


#scope for improvements:
#longer videos --> more chunks to the llm