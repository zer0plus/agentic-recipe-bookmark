from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from agent import RecipeAgent
import uvicorn
import logging
import asyncio
from typing import Optional, Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recipe Extractor", 
    description="Extract recipes from YouTube videos using intelligent LLM agents",
    version="2.1.0"  # Updated version for image gen feature
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_recipe_agent: Optional[RecipeAgent] = None

def get_agent() -> RecipeAgent:
    """Get or create the global recipe agent instance"""
    global _recipe_agent
    if _recipe_agent is None:
        logger.info("Initializing Recipe Agent...")
        _recipe_agent = RecipeAgent()
        logger.info("✅ Recipe Agent initialized successfully")
    return _recipe_agent

class YouTubeRequest(BaseModel):
    youtube_url: HttpUrl
    image_type: Literal["ai", "stock"]  # NEW: Image type selection

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

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.post("/api/extract-recipe", response_model=RecipeResponse)
async def extract_recipe(request: YouTubeRequest):
    try:
        logger.info(f"🚀 Starting recipe extraction from: {request.youtube_url}")
        logger.info(f"🎨 Image type selected: {request.image_type}")
        
        agent = get_agent()
        
        # Pass both URL and image type to agent
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        agent = get_agent()
        return {
            "status": "healthy",
            "agent_initialized": True,
            "tools_count": len(agent.tools),
            "llm_model": "llama3-8b-8192",
            "features": ["ai_image_generation", "stock_images"]  # NEW: Feature list
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/agent-info")
async def agent_info():
    """Get information about the current agent configuration"""
    try:
        agent = get_agent()
        return {
            "architecture": "tool-calling-llm",
            "model": "llama3-8b-8192",
            "tools": [tool.name for tool in agent.tools],
            "capabilities": [
                "dynamic_tool_orchestration",
                "intelligent_workflow_routing", 
                "error_recovery",
                "langsmith_monitoring",
                "ai_image_generation"  # NEW: Added capability
            ],
            "workflow_type": "agentic",
            "description": "Intelligent agent with AI image generation and stock photo capabilities"
        }
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve agent information")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)