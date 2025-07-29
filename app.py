from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, Field
from agent import RecipeAgent
import uvicorn
import logging
import asyncio
from typing import Optional, Literal
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recipe Extractor", 
    description="Extract recipes from YouTube videos using intelligent LLM agents",
    version="0.0.1"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_recipe_agent: Optional[RecipeAgent] = None
_chat_llm: Optional[ChatGroq] = None

def get_agent() -> RecipeAgent:
    """Get or create the global recipe agent instance"""
    global _recipe_agent
    if _recipe_agent is None:
        logger.info("Initializing Recipe Agent...")
        _recipe_agent = RecipeAgent()
        logger.info("✅ Recipe Agent initialized successfully")
    return _recipe_agent

def get_chat_llm() -> ChatGroq:
    """Get or create the chat LLM instance"""
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=300
        )
    return _chat_llm

class YouTubeRequest(BaseModel):
    youtube_url: HttpUrl
    image_type: Literal["ai", "stock"]

class RecipeResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str
    image: str
    prep_time: str
    cook_time: str
    servings: int
    difficulty: str
    ingredients: list[str]
    instructions: list[str]

class ChatRequest(BaseModel):
    recipe_name: str
    ingredients: list[str]
    question: str = Field(max_length=100)
    chat_history: list[dict] = []

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.post("/api/extract-recipe", response_model=RecipeResponse)
async def extract_recipe(request: YouTubeRequest):
    try:
        logger.info(f"🚀 Starting recipe extraction from: {request.youtube_url}")
        logger.info(f"🎨 Image type selected: {request.image_type}")
        
        agent = get_agent()
        
        recipe_data = await agent.extract_recipe_from_youtube(
            str(request.youtube_url), 
            image_type=request.image_type
        )
        
        if not recipe_data:
            logger.error("Agent returned empty recipe data")
            raise HTTPException(
                status_code=422, 
                detail="Could not extract recipe from video - agent returned no data"
            )
        
        logger.info(f"✅ Recipe extraction completed: {recipe_data['name']}")
        logger.info(f"🖼️ Image generated using: {request.image_type} method")
        return recipe_data
        
    except ValueError as e:
        logger.error(f"Validation error during extraction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        logger.error("Recipe extraction timed out")
        raise HTTPException(
            status_code=504, 
            detail="Recipe extraction timed out - video may be too long or unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error during recipe extraction: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during recipe extraction"
        )

@app.post("/api/recipe-chat", response_model=ChatResponse)
async def recipe_chat(request: ChatRequest):
    chat_llm = get_chat_llm()
    ingredients_text = ", ".join(request.ingredients)
    
    history_context = ""
    if request.chat_history:
        recent_history = request.chat_history[-20:]
        for msg in recent_history:
            history_context += f"{msg['type']}: {msg['text']}\n"
    
    prompt = f"""You are a helpful cooking assistant for "{request.recipe_name}".

Ingredients: {ingredients_text}

Previous conversation:
{history_context}

Current question: {request.question}

Give 1-2 short, specific sentences."""

    response = chat_llm.invoke([HumanMessage(content=prompt)])
    return ChatResponse(answer=response.content.strip())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)