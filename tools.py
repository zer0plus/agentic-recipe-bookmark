import random
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re
from tavily import TavilyClient
from langchain_core.tools import tool
import urllib.parse
from youtube_transcript_api.proxies import GenericProxyConfig

@tool
def efficient_tavily_search_tool(query: str, search_type: str = "text", max_results: int = 3) -> dict:
    """
    Optimized Tavily search with result limits for speed.
    
    Args:
        query: Search query string
        search_type: Type of search - "text" for general search, "image" for image search
        max_results: Maximum number of results to return (3 for images, 4 for text recommended)
        
    Returns:
        dict: Search results in format depending on search_type
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {"type": search_type, "found": False, "error": "API key not configured"}
        
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
        if search_type == "image":
            search_results = tavily_client.search(
                query=query,
                search_depth="basic", 
                max_results=max_results,
                include_images=True
            )
            
            if 'images' in search_results and search_results['images']:
                for image_url in search_results['images'][:max_results]:
                    if 'images.unsplash.com' in image_url:
                        formatted_url = f"{image_url.split('?')[0]}?w=900&auto=format&fit=crop&q=60"
                        return {"type": "image", "url": formatted_url, "found": True}
                
                if search_results['images']:
                    return {"type": "image", "url": search_results['images'][0], "found": True}
            
            # Hardcoded fallback images
            fallback_images = [
                "https://images.unsplash.com/photo-1556909114-6bca3ebce58b?w=900&auto=format&fit=crop&q=60",
                    "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=900&auto=format&fit=crop&q=60"
            ]
            return {"type": "image", "url": random.choice(fallback_images), "found": False}
        
        else: 
            search_results = tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_images=False
            )
            
            timing_data = {"prep_time": None, "cook_time": None}
            
            if search_results and 'results' in search_results:
                for result in search_results['results'][:max_results]:
                    text = (result.get('content', '') + " " + result.get('title', '')).lower()
                    
                    if not timing_data["prep_time"]:
                        prep_match = re.search(r'prep(?:aration)?\s*(?:time)?:?\s*(\d+)\s*(?:min|minute)', text)
                        if prep_match:
                            timing_data["prep_time"] = f"{prep_match.group(1)} min"
                    
                    if not timing_data["cook_time"]:
                        cook_match = re.search(r'cook(?:ing)?\s*(?:time)?:?\s*(\d+)\s*(?:min|minute)', text)
                        if cook_match:
                            timing_data["cook_time"] = f"{cook_match.group(1)} min"
            
            result = {"type": "text", "found": True}
            result.update(timing_data)
            return result
            
    except Exception as e:
        return {"type": search_type, "found": False, "error": str(e)}


@tool
def get_transcript_from_youtube_url_tool(youtube_url: str) -> dict:
    """
    Extracts transcript from a YouTube video URL.
    
    Args:
        youtube_url: Full URL of the YouTube video
        
    Returns:
        dict: {
            "video_id": str,  # Extracted YouTube video ID
            "transcript": str  # Cleaned transcript text
        }
        
    Raises:
        ValueError: If URL is invalid or transcript can't be retrieved
    """
    
    try:
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        video_id = None
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                video_id = match.group(1)
                break
        
        if not video_id:
            raise ValueError("Invalid YouTube URL format")
        
        if os.getenv("PROXY_HOST"):
            
            username = urllib.parse.quote(os.getenv('PROXY_USERNAME'), safe='')
            password = urllib.parse.quote(os.getenv('PROXY_PASSWORD'), safe='')
            
            proxy_url = f"http://{username}:{password}@{os.getenv('PROXY_HOST')}:{os.getenv('PROXY_PORT')}"
            
            proxy_config = GenericProxyConfig(
                http_url=proxy_url,
                https_url=proxy_url
            )
            
            ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
        else:
            ytt_api = YouTubeTranscriptApi()
        
        fetched_transcript = ytt_api.fetch(video_id)
        transcript_data = fetched_transcript.to_raw_data()
        
        # Joins complete text together, clean up whitespace and remove special characters
        full_transcript = " ".join([item['text'] for item in transcript_data])
        full_transcript = re.sub(r'\[.*?\]', '', full_transcript)
        full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
        
        if len(full_transcript) > 7000:
            full_transcript = full_transcript[:7000] + "..."
        
        return {
            "video_id": video_id,
            "transcript": full_transcript
        }
        
    except Exception as e:
        raise ValueError(f"Could not retrieve transcript: {e}")