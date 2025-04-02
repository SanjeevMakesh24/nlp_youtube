from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


#function to extract video ID from URL
def extract_video_id(url):

    #https://www.youtube.com/watch?v=042pDj9FJ7Y&ab_channel=TEDxTalks
    #.split("&")[0] to remove any additional parameters
    
    return url.split("v=")[-1].split("&")[0]

#video link
video_link = "https://www.youtube.com/watch?v=042pDj9FJ7Y"  
video_id = extract_video_id(video_link)

try:
    #get transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    #join all text segments
    text = " ".join([entry["text"] for entry in transcript])
    
    # Create a Document object manually
    document = Document(
        page_content=text,
        metadata={"source": f"{video_link}"}
    )

    print(f"Successfully loaded transcript with {len(text)} characters")
    print(document)
    
    
except Exception as e:
    print(f"Error: {e}")