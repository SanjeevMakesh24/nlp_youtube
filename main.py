from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#function to extract video ID from URL
def extract_video_id(url):

    #https://www.youtube.com/watch?v=042pDj9FJ7Y&ab_channel=TEDxTalks
    #.split("&")[0] to remove any additional parameters
    
    return url.split("v=")[-1].split("&")[0]

#video link
video_link = "https://www.youtube.com/watch?v=jLpUEACVBlE"  
video_id = extract_video_id(video_link)

try:
    #get transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    #join all text segments
    text = " ".join([entry["text"] for entry in transcript])

    print(f"Successfully loaded transcript with {len(text)} characters")
    print(text)

    #split the long transcript into smaller chunks
    #chunk_overlap: last 100 characters of one chunk will appear at the beginning of the next chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    print(f"Created {len(docs)} chunks")

    #a sentence-transformer model for embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #FAISS vector store from the chunks
    vectorstore = FAISS.from_documents(docs, embedding_model)


    query = "Who is the largest vehicle exporter?"
    relevent_docs = vectorstore.similarity_search(query, k=3)

    #top matching chunks
    print("\nTop Matching Transcript Chunks:")
    for i, doc in enumerate(relevent_docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)
    
    
except Exception as e:
    print(f"Error: {e}")