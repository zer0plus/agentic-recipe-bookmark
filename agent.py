import os
import re
import random
import json
import time
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

logger = logging.getLogger(__name__)
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "moms-cook-book-streamlined"

class PerformanceTracker:
    def __init__(self):
        self.start_time = None
        self.step_times = {}
        self.llm_calls = 0
        self.tool_calls = 0
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def step(self, step_name: str):
        if self.start_time:
            self.step_times[step_name] = time.time() - self.start_time
            logger.info(f"⏱️ {step_name}: {self.step_times[step_name]:.2f}s")
    
    def llm_call(self):
        self.llm_calls += 1
    
    def tool_call(self):
        self.tool_calls += 1
    
    def summary(self):
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"🏁 Total extraction time: {total_time:.2f}s")
        logger.info(f"📊 LLM calls: {self.llm_calls}, Tool calls: {self.tool_calls}")
        return {
            "total_time": total_time,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "step_times": self.step_times
        }

def rate_limited_llm_call(delay_seconds: float = 0.2):
    """Decorator to add strategic delays between LLM calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Strategic delay before LLM call
            await asyncio.sleep(delay_seconds)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                call_time = time.time() - start_time
                logger.info(f"🤖 LLM call completed in {call_time:.2f}s")
                return result
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    logger.warning(f"⚠️ Rate limit hit, backing off...")
                    await asyncio.sleep(delay_seconds * 2)  # Exponential backoff
                    return await func(*args, **kwargs)
                raise e
        return wrapper
    return decorator

def keep_first_value(current, new):
    if current is not None and current != '':
        return current
    return new if new is not None else current

def keep_latest_value(current, new):
    return new if new is not None else current

class RecipeState(TypedDict):
    messages: Annotated[list, add_messages]
    youtube_url: Annotated[str, keep_first_value]
    video_id: Annotated[Optional[str], keep_first_value]
    transcript: Annotated[Optional[str], keep_first_value]
    raw_recipe: Annotated[Optional[dict], keep_first_value]
    validated_recipe: Annotated[Optional[dict], keep_latest_value]
    image_url: Annotated[Optional[str], keep_latest_value]
    final_recipe: Annotated[Optional[dict], keep_latest_value]
    current_step: Annotated[str, keep_latest_value]
    error: Annotated[Optional[str], keep_latest_value]
    performance: Annotated[Optional[dict], keep_latest_value]
    search_decisions: Annotated[Optional[dict], keep_latest_value]

class Recipe(BaseModel):
    name: str
    description: str
    category: str
    prep_time: str
    cook_time: str
    servings: int
    difficulty: str
    ingredients: List[str]
    instructions: List[str]


@tool
def get_transcript_from_youtube_url_tool(youtube_url: str) -> dict:
    """Extract video ID and get transcript efficiently"""
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
        
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        transcript_data = fetched_transcript.to_raw_data()
        
        full_transcript = " ".join([item['text'] for item in transcript_data])
        full_transcript = re.sub(r'\[.*?\]', '', full_transcript)
        full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
        
        # Truncate for efficiency
        if len(full_transcript) > 8000:
            full_transcript = full_transcript[:8000] + "..."
        
        logger.info(f"🎬 Video ID: {video_id}, Transcript: {len(full_transcript)} chars")
        
        return {
            "video_id": video_id,
            "transcript": full_transcript
        }
        
    except Exception as e:
        logger.error(f"Transcript extraction failed: {e}")
        raise ValueError(f"Could not retrieve transcript: {e}")

@tool
def efficient_tavily_search_tool(query: str, search_type: str = "text", max_results: int = 3) -> dict:
    """
    Optimized Tavily search with result limits for speed
    
    FOR IMAGE SEARCH: search_type="image", max_results=3
    FOR TIMING DATA: search_type="text", max_results=4
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {"type": search_type, "found": False, "error": "API key not configured"}
        
        tavily_client = TavilyClient(api_key=tavily_api_key)
        logger.info(f"🔍 Fast search: {query[:50]}... (type: {search_type})")
        
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
                        logger.info(f"✅ Found image: {formatted_url}")
                        return {"type": "image", "url": formatted_url, "found": True}
                
                if search_results['images']:
                    return {"type": "image", "url": search_results['images'][0], "found": True}
            
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
                for result in search_results['results'][:max_results]:  # Limit processing
                    text = (result.get('content', '') + " " + result.get('title', '')).lower()
                    
                    if not timing_data["prep_time"]:
                        prep_match = re.search(r'prep(?:aration)?\s*(?:time)?:?\s*(\d+)\s*(?:min|minute)', text)
                        if prep_match:
                            timing_data["prep_time"] = f"{prep_match.group(1)} min"
                    
                    if not timing_data["cook_time"]:
                        cook_match = re.search(r'cook(?:ing)?\s*(?:time)?:?\s*(\d+)\s*(?:min|minute)', text)
                        if cook_match:
                            timing_data["cook_time"] = f"{cook_match.group(1)} min"
            
            logger.info(f"⏱️ Found timing: {timing_data}")
            result = {"type": "text", "found": True}
            result.update(timing_data)
            return result
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"type": search_type, "found": False, "error": str(e)}

class RecipeAgent:
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        
        # Initialize LangSmith client for debugging and tracking
        self.langsmith_client = Client()
        
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=1500,
            callbacks=[LangChainTracer()]
        ).bind_tools([efficient_tavily_search_tool])
        
        self.tools = {
            'get_transcript_from_url': get_transcript_from_youtube_url_tool,
        }
        
        # Tools list for ToolNode (only search tools for visible graph)
        self.search_tools = [efficient_tavily_search_tool]
        
        # Counter for unique IDs
        self._id_counter = 1000
        
        # Build optimized graph
        self.app = self._build_streamlined_graph()

    def _build_streamlined_graph(self) -> StateGraph:
        """Build clean, streamlined LangGraph workflow"""
        workflow = StateGraph(RecipeState)
        
        workflow.add_node("get_transcript", self._get_transcript_node)
        workflow.add_node("parse_recipe", self._parse_recipe_node)
        workflow.add_node("intelligent_enhance", self._intelligent_enhance_node)
        workflow.add_node("search_tool", ToolNode(self.search_tools))  # Visible search node
        workflow.add_node("format_output", self._format_output_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        workflow.add_edge(START, "get_transcript")
        workflow.add_edge("get_transcript", "parse_recipe")
        workflow.add_edge("parse_recipe", "intelligent_enhance")
        
        workflow.add_conditional_edges(
            "intelligent_enhance",
            self._should_use_search_tools,
            {
                "use_search": "search_tool",
                "skip_search": "format_output",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("search_tool", "format_output")
        workflow.add_edge("format_output", END)
        workflow.add_edge("handle_error", END)
        
        workflow.add_conditional_edges(
            "get_transcript",
            lambda state: "handle_error" if state.get("error") else "parse_recipe"
        )
        workflow.add_conditional_edges(
            "parse_recipe", 
            lambda state: "handle_error" if state.get("error") else "intelligent_enhance"
        )
        
        return workflow.compile()

    def _should_use_search_tools(self, state: RecipeState) -> str:
        """Determine if search tools should be used based on LLM decisions"""
        if state.get("error"):
            return "error"
        
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            return "use_search"
        else:
            return "skip_search"

    def _get_transcript_node(self, state: RecipeState) -> RecipeState:
        """Fast transcript extraction"""
        try:
            self.performance_tracker.step("transcript_start")
            logger.info("🎬 Getting transcript...")
            
            result = self.tools['get_transcript_from_url'].invoke({"youtube_url": state["youtube_url"]})
            
            state["video_id"] = result["video_id"]
            state["transcript"] = result["transcript"]
            state["current_step"] = "transcript_complete"
            
            self.performance_tracker.step("transcript_complete")
            
        except Exception as e:
            state["error"] = f"Transcript failed: {str(e)}"
            logger.error(f"❌ Transcript error: {e}")
        
        return state

    @rate_limited_llm_call(delay_seconds=0.2)
    async def _llm_parse_recipe(self, transcript: str) -> str:
        """Optimized recipe parsing with ultra-concise prompt"""
        self.performance_tracker.llm_call()
        
        prompt = f"""Extract recipe JSON from transcript. Return ONLY valid JSON:

TRANSCRIPT: {transcript[:6000]}

FORMAT:
{{"name":"Recipe Name","description":"Brief description","category":"Main Course|Appetizer|Dessert|Drink|Side Dish","prep_time":"X min","cook_time":"X min","servings":number,"difficulty":"Easy|Medium|Hard","ingredients":["ingredient 1"],"instructions":["step 1"]}}

Use "undefined" for missing timing. JSON only:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _parse_recipe_node(self, state: RecipeState) -> RecipeState:
        """Streamlined recipe parsing"""
        try:
            self.performance_tracker.step("parse_start")
            logger.info("🍳 Parsing recipe...")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            recipe_text = loop.run_until_complete(self._llm_parse_recipe(state["transcript"]))
            loop.close()
            
            try:
                recipe_data = json.loads(recipe_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', recipe_text, re.DOTALL)
                if json_match:
                    clean_json = json_match.group()
                    clean_json = re.sub(r',(\s*[}\]])', r'\1', clean_json)
                    recipe_data = json.loads(clean_json)
                else:
                    raise ValueError("Could not parse recipe JSON")
            
            state["raw_recipe"] = recipe_data
            state["current_step"] = "parse_complete"
            
            self.performance_tracker.step("parse_complete")
            logger.info(f"✅ Parsed: {recipe_data.get('name', 'Unknown')}")
            
        except Exception as e:
            state["error"] = f"Parse failed: {str(e)}"
            logger.error(f"❌ Parse error: {e}")
        
        return state

    @rate_limited_llm_call(delay_seconds=0.2)
    async def _llm_intelligent_decisions(self, recipe: dict) -> tuple:
        """LLM makes decisions and generates tool calls if needed"""
        self.performance_tracker.llm_call()
        
        # Check what's missing
        recipe_name = recipe.get('name', 'Unknown Recipe')
        prep_missing = not recipe.get('prep_time') or recipe.get('prep_time') == 'undefined'
        cook_missing = not recipe.get('cook_time') or recipe.get('cook_time') == 'undefined'
        needs_timing = prep_missing or cook_missing
        needs_image = True  # Always search for image
        
        if needs_timing or needs_image:
            # Create tool calls for the LLM to generate
            prompt = f"""You need to search for missing recipe data. Generate appropriate tool calls.

Recipe: {recipe_name}
Missing timing: {needs_timing}
Needs image: {needs_image}

Call efficient_tavily_search_tool for:
1. If timing missing: query="{recipe_name} prep time cook time", search_type="text", max_results=3
2. For image: query="unsplash {recipe_name} food recipe", search_type="image", max_results=2

Generate the tool calls now."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response, {"needs_timing": needs_timing, "needs_image": needs_image, "recipe_name": recipe_name}
        else:
            # No tools needed
            return None, {"needs_timing": False, "needs_image": False, "recipe_name": recipe_name}

    def _intelligent_enhance_node(self, state: RecipeState) -> RecipeState:
        """Generate tool calls for enhancement if needed"""
        try:
            self.performance_tracker.step("enhance_start")
            logger.info("🧠 Intelligent enhancement...")
            
            raw_recipe = state["raw_recipe"]
            
            # LLM decides what to search for and generates tool calls
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llm_response, decisions = loop.run_until_complete(self._llm_intelligent_decisions(raw_recipe))
            loop.close()
            
            # Store decisions for later use
            state["search_decisions"] = decisions
            
            if llm_response and hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
                # Add the LLM response with tool calls to messages
                state["messages"] = state.get("messages", []) + [llm_response]
                logger.info(f"🔧 Generated {len(llm_response.tool_calls)} tool calls")
            else:
                # No tools needed, proceed directly
                logger.info("✅ No enhancement needed")
            
            state["current_step"] = "enhance_complete"
            self.performance_tracker.step("enhance_complete")
            
        except Exception as e:
            state["error"] = f"Enhancement failed: {str(e)}"
            logger.error(f"❌ Enhancement error: {e}")
        
        return state

    def _apply_search_results(self, messages: list, recipe: dict) -> tuple:
        """Apply search results from messages to recipe and extract image URL"""
        enhanced_recipe = recipe.copy()
        image_url = None
        
        # Look for tool messages in the conversation
        for message in messages:
            if hasattr(message, 'content') and message.content:
                try:
                    # ToolNode puts results in message content
                    if isinstance(message.content, list):
                        # Multiple tool results
                        for result in message.content:
                            if isinstance(result, dict):
                                enhanced_recipe, image_url = self._apply_single_result(result, enhanced_recipe, image_url)
                    elif isinstance(message.content, str):
                        # Single result as string, try to parse
                        try:
                            result = json.loads(message.content)
                            enhanced_recipe, image_url = self._apply_single_result(result, enhanced_recipe, image_url)
                        except:
                            continue
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
        
        return enhanced_recipe, image_url

    def _apply_single_result(self, result: dict, recipe: dict, current_image_url: str) -> tuple:
        """Apply a single search result to the recipe"""
        if result.get('type') == 'text' and result.get('found'):
            # Timing data
            if result.get('prep_time') and (not recipe.get('prep_time') or recipe.get('prep_time') == 'undefined'):
                recipe['prep_time'] = result['prep_time']
                logger.info(f"✅ Updated prep_time: {result['prep_time']}")
            
            if result.get('cook_time') and (not recipe.get('cook_time') or recipe.get('cook_time') == 'undefined'):
                recipe['cook_time'] = result['cook_time']
                logger.info(f"✅ Updated cook_time: {result['cook_time']}")
        
        elif result.get('type') == 'image':
            # Image data
            if result.get('url') and not current_image_url:
                current_image_url = result['url']
                logger.info(f"✅ Found image: {result['url']}")
        
        return recipe, current_image_url

    def _format_output_node(self, state: RecipeState) -> RecipeState:
        """Smart output formatting that handles both search and no-search scenarios"""
        try:
            self.performance_tracker.step("format_start")
            logger.info("🎨 Formatting output...")
            
            raw_recipe = state["raw_recipe"]
            enhanced_recipe = raw_recipe.copy()
            image_url = None
            
            # Check if we have search results to apply
            messages = state.get("messages", [])
            if messages:
                enhanced_recipe, image_url = self._apply_search_results(messages, raw_recipe)
            
            # Set fallback image if none found
            if not image_url:
                fallback_images = [
                    "https://images.unsplash.com/photo-1556909114-6bca3ebce58b?w=900&auto=format&fit=crop&q=60",
                    "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=900&auto=format&fit=crop&q=60",
                    "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=900&auto=format&fit=crop&q=60"
                ]
                image_url = random.choice(fallback_images)
                logger.info("🖼️ Using fallback image")
            
            # Fast validation with Pydantic
            try:
                recipe = Recipe(**enhanced_recipe)
                validated_recipe = recipe.model_dump()
            except Exception as e:
                logger.warning(f"Validation warning: {e}")
                validated_recipe = enhanced_recipe
            
            # Create final formatted output
            final_recipe = {
                "id": self._id_counter,
                "name": validated_recipe["name"],
                "description": validated_recipe["description"],
                "category": validated_recipe["category"], 
                "image": image_url,
                "prep_time": validated_recipe["prep_time"],
                "cook_time": validated_recipe["cook_time"],
                "servings": validated_recipe["servings"],
                "difficulty": validated_recipe["difficulty"],
                "ingredients": validated_recipe["ingredients"],
                "instructions": validated_recipe["instructions"]
            }
            
            state["validated_recipe"] = validated_recipe
            state["image_url"] = image_url
            state["final_recipe"] = final_recipe
            state["current_step"] = "completed"
            state["performance"] = self.performance_tracker.summary()
            
            self._id_counter += 1
            
            self.performance_tracker.step("format_complete")
            logger.info(f"🎉 Complete: {final_recipe['name']}")
            
        except Exception as e:
            state["error"] = f"Format failed: {str(e)}"
            logger.error(f"❌ Format error: {e}")
        
        return state

    def _handle_error_node(self, state: RecipeState) -> RecipeState:
        """Streamlined error handling"""
        error_msg = state.get('error', 'Unknown error')
        logger.error(f"🚨 Pipeline failed: {error_msg}")
        state["performance"] = self.performance_tracker.summary()
        return state

    async def extract_recipe_from_youtube(self, youtube_url: str) -> Optional[Dict]:
        """Main extraction method - optimized for 8-12 second target"""
        try:
            # Start performance tracking
            self.performance_tracker.start()
            logger.info(f"🚀 Starting streamlined extraction: {youtube_url}")
            
            initial_state = {
                "messages": [],
                "youtube_url": youtube_url,
                "video_id": None,
                "transcript": None,
                "raw_recipe": None,
                "validated_recipe": None,
                "image_url": None,
                "final_recipe": None,
                "current_step": "start",
                "error": None,
                "performance": None,
                "search_decisions": None
            }
            
            config = {"configurable": {"thread_id": f"recipe-{hash(youtube_url)}"}}
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            if final_state.get("error"):
                logger.error(f"❌ Pipeline failed: {final_state['error']}")
                raise ValueError(final_state["error"])
            
            if final_state.get("final_recipe"):
                recipe = final_state["final_recipe"]
                perf = final_state.get("performance", {})
                decisions = final_state.get("search_decisions", {})
                
                logger.info(f"🎉 Streamlined extraction complete:")
                logger.info(f"   📝 {recipe['name']}")
                logger.info(f"   ⏱️ {recipe.get('prep_time')} | {recipe.get('cook_time')}")
                logger.info(f"   🔍 Timing searched: {decisions.get('needs_timing', False)}")
                logger.info(f"   🖼️ Image searched: {decisions.get('needs_image', False)}")
                logger.info(f"   📊 Total time: {perf.get('total_time', 0):.2f}s")
                
                return recipe
            else:
                raise ValueError("No recipe extracted")
            
        except Exception as e:
            logger.error(f"💥 Extraction failed: {e}")
            self.performance_tracker.summary()
            raise e

# === LANGGRAPH STUDIO COMPATIBILITY ===
def create_graph():
    """Factory function for LangGraph Studio"""
    return RecipeAgent().app

# For LangGraph Studio
graph = create_graph()