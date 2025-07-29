import os
import re
import json
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated
from tools import get_transcript_from_youtube_url_tool, efficient_tavily_search_tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "moms-cook-book-streamlined"

def rate_limited_llm_call(delay_seconds: float = 0.2):
    """Decorator to add strategic delays between LLM calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Delay before LLM call
            await asyncio.sleep(delay_seconds)
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                if "rate_limit" in str(e).lower():
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
    image_type: Annotated[str, keep_first_value]
    video_id: Annotated[Optional[str], keep_first_value]
    transcript: Annotated[Optional[str], keep_first_value]
    raw_recipe: Annotated[Optional[dict], keep_first_value]
    validated_recipe: Annotated[Optional[dict], keep_latest_value]
    image_url: Annotated[Optional[str], keep_latest_value]
    final_recipe: Annotated[Optional[dict], keep_latest_value]
    current_step: Annotated[str, keep_latest_value]
    error: Annotated[Optional[str], keep_latest_value]
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

class RecipeAgent:
    def __init__(self):
        
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
        
        self.search_tools = [efficient_tavily_search_tool]
        
        # Counter for unique IDs
        self._id_counter = 1000
        
        self.app = self._build_graph()

    def _preprocess_json_text(self, json_text: str) -> str:
        """Fix common JSON issues before parsing"""
        def fix_servings_range(match):
            range_str = match.group(1)
            if '-' in range_str:
                start, end = map(int, range_str.split('-'))
                avg = (start + end) // 2
                return f'"servings": {avg}'
            return match.group(0)
        
        # Fix servings field specifically
        json_text = re.sub(r'"servings":\s*([0-9-]+)', fix_servings_range, json_text)
        
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)  # Remove trailing commas
        json_text = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1, \2', json_text)  # Fix line breaks
        
        return json_text

    def generate_ai_image(self, recipe_name: str) -> str:
        """Generate AI image using Pollinations API"""
        import urllib.parse
        prompt = f"professional food photography of {recipe_name}, appetizing, well-lit, high quality"
        encoded_prompt = urllib.parse.quote(prompt)
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=900&height=600&nologo=true"

    def _build_graph(self) -> StateGraph:
        """Build clean, streamlined LangGraph workflow"""
        workflow = StateGraph(RecipeState)
        
        workflow.add_node("get_transcript", self._get_transcript_node)
        workflow.add_node("parse_recipe", self._parse_recipe_node)
        workflow.add_node("intelligent_enhance", self._intelligent_enhance_node)
        workflow.add_node("search_tool", ToolNode(self.search_tools))
        workflow.add_node("generate_recipe_img", self._generate_recipe_img_node)
        workflow.add_node("format_output", self._format_output_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        workflow.add_edge(START, "get_transcript")
        workflow.add_edge("get_transcript", "parse_recipe")
        workflow.add_edge("parse_recipe", "intelligent_enhance")
        
        workflow.add_conditional_edges(
            "intelligent_enhance",
            self._route_after_enhance,
            {
                "use_search": "search_tool",
                "generate_ai_image": "generate_recipe_img", 
                "skip_to_format": "format_output",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "search_tool",
            lambda state: "generate_recipe_img" if state.get("image_type") == "ai" else "format_output"
        )
        
        workflow.add_edge("generate_recipe_img", "format_output")
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

    def _route_after_enhance(self, state: RecipeState) -> str:
        """Route to appropriate next step based on image type and search needs"""
        if state.get("error"):
            return "error"
        
        messages = state.get("messages", [])
        image_type = state.get("image_type", "stock")
        
        # If LLM generated tool calls for search, use search
        if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            return "use_search"
        
        # If no search needed but need AI image generation
        elif image_type == "ai":
            return "generate_ai_image"
        
        else:
            return "skip_to_format"

    def _get_transcript_node(self, state: RecipeState) -> RecipeState:
        """Fast transcript extraction"""
        try:
            result = self.tools['get_transcript_from_url'].invoke({"youtube_url": state["youtube_url"]})
            
            state["video_id"] = result["video_id"]
            state["transcript"] = result["transcript"]
            state["current_step"] = "transcript_complete"
            
        except Exception as e:
            state["error"] = f"Transcript failed: {str(e)}"
        
        return state

    def _generate_recipe_img_node(self, state: RecipeState) -> RecipeState:
        """Generate AI image for recipe"""
        try:
            raw_recipe = state["raw_recipe"]
            if not raw_recipe:
                state["error"] = "Cannot generate image - no recipe data available"
                return state
            
            recipe_name = raw_recipe.get('name', 'recipe')
            ai_image_url = self.generate_ai_image(recipe_name)
            state["image_url"] = ai_image_url
            state["current_step"] = "image_generated"
            
        except Exception as e:
            state["error"] = f"Image generation failed: {str(e)}"
        
        return state

    @rate_limited_llm_call(delay_seconds=0.2)
    async def _llm_parse_recipe(self, transcript: str) -> str:
        """Optimized recipe parsing with ultra-concise prompt"""
        
        prompt = f"""Extract recipe JSON from transcript. Return ONLY valid JSON:

TRANSCRIPT: {transcript[:7000]}

FORMAT:
{{"name":"Recipe Name","description":"Brief description","category":"Main Course|Appetizer|Dessert|Drink|Side Dish","prep_time":"X min","cook_time":"X min","servings":number,"difficulty":"Easy|Medium|Hard","ingredients":["ingredient 1"],"instructions":["step 1"]}}

Use "undefined" for missing timing. JSON only:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _parse_recipe_node(self, state: RecipeState) -> RecipeState:
        """Streamlined recipe parsing"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            recipe_text = loop.run_until_complete(self._llm_parse_recipe(state["transcript"]))
            loop.close()
            
            recipe_text = self._preprocess_json_text(recipe_text)
            
            try:
                recipe_data = json.loads(recipe_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', recipe_text, re.DOTALL)
                if json_match:
                    clean_json = json_match.group()
                    clean_json = re.sub(r',(\s*[}\]])', r'\1', clean_json)
                    clean_json = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1, \2', clean_json)
                    try:
                        recipe_data = json.loads(clean_json)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Could not parse recipe JSON after cleanup: {e}")
                else:
                    raise ValueError("Could not find JSON structure in LLM response")
            
            state["raw_recipe"] = recipe_data
            state["current_step"] = "parse_complete"
            
        except Exception as e:
            state["error"] = f"Parse failed: {str(e)}"
        
        return state

    @rate_limited_llm_call(delay_seconds=0.2)
    async def _llm_intelligent_decisions(self, recipe: dict, image_type: str) -> tuple:
        """LLM makes decisions and generates tool calls if needed"""
        
        # Check what's missing
        recipe_name = recipe.get('name', 'Unknown Recipe')
        prep_missing = not recipe.get('prep_time') or recipe.get('prep_time') == 'undefined'
        cook_missing = not recipe.get('cook_time') or recipe.get('cook_time') == 'undefined'
        needs_timing = prep_missing or cook_missing
        needs_image_search = image_type == "stock"
        
        if needs_timing or needs_image_search:
            # Create tool calls for the LLM to generate
            tool_instructions = []
        
            if needs_timing:
                tool_instructions.append(f'1. For timing: query="{recipe_name} prep time cook time", search_type="text", max_results=3')
            
            if needs_image_search:
                tool_instructions.append(f'2. For image: query="unsplash {recipe_name} food recipe", search_type="image", max_results=2')
            
            prompt = f"""You need to search for missing recipe data. Generate appropriate tool calls.

Recipe: {recipe_name}
Missing timing: {needs_timing}
Image type: {image_type} {"(will search)" if needs_image_search else "(will use AI generation)"}

Call efficient_tavily_search_tool for:
{chr(10).join(tool_instructions)}

Generate the tool calls now."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response, {"needs_timing": needs_timing, "needs_image_search": needs_image_search, "recipe_name": recipe_name, "image_type": image_type}
        else:
            # No tools needed
            return None, {"needs_timing": False, "needs_image_search": False, "recipe_name": recipe_name, "image_type": image_type}

    def _intelligent_enhance_node(self, state: RecipeState) -> RecipeState:
        """Generate tool calls for enhancement if needed"""
        try:
            raw_recipe = state["raw_recipe"]
            if not raw_recipe:
                state["error"] = "Cannot enhance - recipe parsing failed"
                return state
                
            image_type = state.get("image_type", "stock")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llm_response, decisions = loop.run_until_complete(self._llm_intelligent_decisions(raw_recipe, image_type))
            loop.close()
            
            state["search_decisions"] = decisions
            
            if llm_response and hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
                state["messages"] = state.get("messages", []) + [llm_response]
            
            state["current_step"] = "enhance_complete"
            
        except Exception as e:
            state["error"] = f"Enhancement failed: {str(e)}"
        
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
                    continue
        
        return enhanced_recipe, image_url

    def _apply_single_result(self, result: dict, recipe: dict, current_image_url: str) -> tuple:
        """Apply a single search result to the recipe"""
        if result.get('type') == 'text' and result.get('found'):
            if result.get('prep_time') and (not recipe.get('prep_time') or recipe.get('prep_time') == 'undefined'):
                recipe['prep_time'] = result['prep_time']
            
            if result.get('cook_time') and (not recipe.get('cook_time') or recipe.get('cook_time') == 'undefined'):
                recipe['cook_time'] = result['cook_time']
        
        elif result.get('type') == 'image':
            if result.get('url') and not current_image_url:
                current_image_url = result['url']
        
        return recipe, current_image_url

    def _format_output_node(self, state: RecipeState) -> RecipeState:
        """Smart output formatting that handles both search and no-search scenarios"""
        try:
            raw_recipe = state["raw_recipe"]
            if not raw_recipe:
                state["error"] = "Cannot format - no recipe data available"
                return state
                
            enhanced_recipe = raw_recipe.copy()
            image_url = state.get("image_url")  # Check if already generated by AI node
            image_type = state.get("image_type", "stock")
            
            messages = state.get("messages", [])
            if messages:
                enhanced_recipe, search_image_url = self._apply_search_results(messages, raw_recipe)
                # Use search result image if found, otherwise keep existing image_url
                if search_image_url:
                    image_url = search_image_url
            
            try:
                recipe = Recipe(**enhanced_recipe)
                validated_recipe = recipe.model_dump()
            except Exception as e:
                validated_recipe = enhanced_recipe.copy()
                
                defaults = {
                    "name": "Extracted Recipe",
                    "description": "Recipe extracted from video",
                    "category": "Main Course",
                    "prep_time": "15 min",
                    "cook_time": "30 min", 
                    "servings": 4,
                    "difficulty": "Medium",
                    "ingredients": ["Ingredients not fully extracted"],
                    "instructions": ["Instructions not fully extracted"]
                }
                
                for key, default_value in defaults.items():
                    if key not in validated_recipe or not validated_recipe[key]:
                        validated_recipe[key] = default_value
            
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
            
            self._id_counter += 1
            
        except Exception as e:
            state["error"] = f"Format failed: {str(e)}"
        
        return state

    def _handle_error_node(self, state: RecipeState) -> RecipeState:
        """Streamlined error handling"""
        error_msg = state.get('error', 'Unknown error')
        return state

    async def extract_recipe_from_youtube(self, youtube_url: str, image_type: str = "stock") -> Optional[Dict]:
        """Main extraction method with AI image generation support"""
        try:
            
            initial_state = {
                "messages": [],
                "youtube_url": youtube_url,
                "image_type": image_type,
                "video_id": None,
                "transcript": None,
                "raw_recipe": None,
                "validated_recipe": None,
                "image_url": None,
                "final_recipe": None,
                "current_step": "start",
                "error": None,
                "search_decisions": None
            }
            
            config = {"configurable": {"thread_id": f"recipe-{hash(youtube_url)}"}}
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            if final_state.get("error"):
                raise ValueError(final_state["error"])
            
            if final_state.get("final_recipe"):
                recipe = final_state["final_recipe"]
                return recipe
            else:
                raise ValueError("No recipe extracted")
            
        except Exception as e:
            raise e

def create_graph():
    """Factory function for LangGraph Studio"""
    return RecipeAgent().app

graph = create_graph()