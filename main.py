from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


#function to extract video ID from URL
def extract_video_id(url):

    #https://www.youtube.com/watch?v=042pDj9FJ7Y&ab_channel=TEDxTalks
    #.split("&")[0] to remove any additional parameters
    
    return url.split("v=")[-1].split("&")[0]

#video link
video_link = "https://www.youtube.com/watch?v=jmrHBHJTVZ4&t=2492s"  
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


    query = "What do they talk about Apple products, ipads, iphones?"
    relevent_docs = vectorstore.similarity_search(query, k=3) #list

    #print(type(relevent_docs))

    #top matching chunks
    print("\nTop Matching Transcript Chunks:")
    for i, doc in enumerate(relevent_docs):
        print(f"\n--- Chunk {i+1} ---")
        timestamp = doc.metadata.get("timestamp", "N/A")
        print(f"[Timestamp: {timestamp:.2f} sec]")
        print(doc.page_content)

    

except Exception as e:
    print(f"Error: {e}")