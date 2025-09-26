import sys
import json

from typing import Optional, Type, Dict, Any, Union, Literal
import ast
import unicodedata

from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field, field_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from typing import List

from langchain_community.utilities.serpapi import SerpAPIWrapper

import pickle
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from okg.load_and_embed import customized_trend_retriever
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

class CounterInput(BaseModel):
    in_str: str = Field(description="A List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]."\
                        "Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy.")
    
    @field_validator('in_str', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class CustomCounterTool(BaseTool):
    name: str = "character_counter"
    description: str = "A character counter. Useful for counting the number of characters in a sentence. Takes as input a List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]. \
        Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False


    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        """Returns the number of characters in each input sentence."""
        return_str = ""
        for sent in in_sent:
            c_count = count_chars(sent[0])
            limit = int(sent[1])
            return_str += f"{sent[0]}: {c_count}/{sent[1]} characters"
            if c_count > limit:
                return_str += " (Too long)\n"
            elif c_count < limit//2:
                return_str += " (Too short)\n"
            else:
                return_str += "\n"
        return return_str

def count_chars(s):
    count = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ['F', 'W']:  # Full-width or Wide characters
            count += 2
        else:
            count += 1
    return count


class SerpAPIInput(BaseModel):
    in_str: str = Field(description="Input as String")

    @field_validator('in_str', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class SerpAPITool(BaseTool):
    name: str = "google_search"
    description: str = "A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        searched_dict = retrieveSearches()
        if in_str in searched_dict:
            print("\nNote: Loaded from backup")
            return searched_dict[in_str]
        
        search_res = SerpAPIWrapper().run(in_str)
        searched_dict[in_str] = search_res
        saveSearches(searched_dict)
        return search_res



class OutputTool(BaseTool):
    name: str = "output_tool"
    description: str = "A tool to simply write your thoughts. Nothing will be return for output."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return ""


class ClickAggregator(BaseTool):
    name: str = "click_aggregator"
    description: str = "Returns the total number of clicks per category for the current ad setting."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    click_df: pd.DataFrame = None

    def __init__(self, file):
        super().__init__()
        self.click_df = pd.read_csv(file)

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        aggr_clicks = self.click_df.groupby("Category", as_index=False).sum()
        average = aggr_clicks["Clicks"].mean()
        return_text = ""
        for row in range(len(aggr_clicks)):
            return_text += f"Category: {aggr_clicks['Category'][row]}\nClicks: {aggr_clicks['Clicks'][row]}\nDifference to Average: {aggr_clicks['Clicks'][row] - average: .3f}\n\n"
        return return_text



from PIL import Image
import base64

class ImageInput(BaseModel):
    path: str  # Path to the image file

class ImageToBase64Tool(BaseTool):
    name: str = "image_to_base64"
    description: str = "Encodes an image file to a base64 data URI for use in vision models."
    args_schema: Type[BaseModel] = ImageInput
    return_direct: bool = False


    def _run(
        self,
        path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Clean up path - remove sandbox prefix and normalize
        cleaned_path = path
        
        # Remove common sandbox or invalid prefixes
        prefixes_to_remove = ["sandbox:", "sandbox:/mnt/data/", "/mnt/data/"]
        for prefix in prefixes_to_remove:
            if cleaned_path.startswith(prefix):
                cleaned_path = cleaned_path[len(prefix):]
                print(f"🔧 Cleaned path: removed '{prefix}' -> {cleaned_path}")
        
        # Normalize path and make it absolute if it's relative
        if not os.path.isabs(cleaned_path):
            # If it's a relative path, try to resolve it relative to current working directory
            cleaned_path = os.path.abspath(cleaned_path)
            print(f"🔧 Converted to absolute path: {cleaned_path}")
        
        # Check if file exists
        if not os.path.exists(cleaned_path):
            # If the absolute path doesn't exist, try some common fallback locations
            possible_paths = [
                path,  # Original path as provided
                os.path.join("outputs", os.path.basename(cleaned_path)),  # In outputs directory
                os.path.join("outputs/competitive_test", os.path.basename(cleaned_path)),  # In competitive test directory
                os.path.join(os.getcwd(), "outputs", os.path.basename(cleaned_path)),  # Absolute outputs directory
            ]
            
            for test_path in possible_paths:
                if os.path.exists(test_path):
                    cleaned_path = test_path
                    print(f"🎯 Found image at: {cleaned_path}")
                    break
            else:
                # List available files for debugging
                output_dirs = ["outputs", "outputs/competitive_test", "."]
                available_files = []
                for dir_path in output_dirs:
                    if os.path.exists(dir_path):
                        try:
                            files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                            if files:
                                available_files.extend([f"{dir_path}/{f}" for f in files])
                        except PermissionError:
                            continue
                
                error_msg = f"Error: Image file not found at {path}"
                if cleaned_path != path:
                    error_msg += f" (cleaned: {cleaned_path})"
                if available_files:
                    error_msg += f"\n📁 Available image files: {available_files[:10]}"  # Show first 10 files
                return error_msg
        
        try:
            with open(cleaned_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                data_uri = f"data:image/png;base64,{base64_image}"
                print(f"✅ Successfully encoded image: {cleaned_path}")
                # 🎯 FIX: Return as JSON string instead of dict to avoid parsing issues
                return json.dumps({"type": "image_url", "image_url": {"url": data_uri}})
        except Exception as e:
            return f"Error encoding image at {cleaned_path}: {str(e)}"



from openai import AzureOpenAI
from typing import Optional, Type, Dict, Any
import json
import requests
# class Dalle3GenInput(BaseModel):
#     prompt: str
#     filepath: Optional[str] = "../output/generated_image.png"

# class Dalle3ImageGenTool(BaseTool):
#     name: str = "dalle3_image_generator"
#     description: str = "Generates an image using Azure OpenAI DALL·E 3 and saves it locally."
#     args_schema: Type[BaseModel] = Dalle3GenInput
#     return_direct: bool = False
#     llm: AzureOpenAI = None
#     folder: str = "../output"
#     def __init__(self, config):
#         super().__init__()
#         self.llm = AzureOpenAI(
#             api_version=config["LLM"]["openai_dalle_version"],
#             api_key=config["KEYS"]["OPENAI_DALLE_API_KEY"],
#             azure_endpoint=config["KEYS"]["OPENAI_DALLE_ENDPOINT"]
#         )
#         self.folder = config["SETTING"]["output_folder"]
#     def _run(
#         self,
#         prompt: str,
#         filepath: str = "../output/generated_image.png",
#         run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> Dict[str, Any]:
#         if "EXP" not in filepath:
#             raise "Filepath is incorrect."
#         print(f"--- Running Dalle3ImageGenTool ---")
#         try:
#             # Generate the image
#             print("Generating image...")
#             result = self.llm.images.generate(
#                 model="dall-e-3",
#                 prompt=prompt,
#                 n=1
#             )
#             json_response = json.loads(result.model_dump_json())
#             image_url = json_response["data"][0]["url"]
#             print(f"Image URL: {image_url}")

#             # Save locally
#             os.makedirs(os.path.dirname(filepath), exist_ok=True)

#             image_bytes = requests.get(image_url).content
#             with open(filepath, "wb") as f:
#                 f.write(image_bytes)
#             print(f"Image saved at {filepath}")

#             # Summary
#             text_summary = (
#                 f"✅ Image generated and saved at: `{filepath}`\n"
#                 f"📷 Prompt used:\n```text\n{prompt}\n```"
#             )

#             return {
#                 "text_summary": text_summary,
#                 "image_path": filepath
#             }

#         except Exception as e:
#             print(f"❌ Error: {repr(e)}", file=sys.stderr)
#             return {
#                 "text_summary": f"❌ Failed to generate image. Error: {repr(e)}",
#                 "image_path": None
#             }

from openai import OpenAI
from typing import Literal
class RecraftGenInput(BaseModel):
    prompt: str
    size: Literal["1024x1024"] = "1024x1024"  # 🎯 FIXED: Only allow configured banner size to prevent agent from choosing different sizes
    # filepath: Optional[str] = "../output/generated_image.png"

class RecraftImageGenTool(BaseTool):
    name: str = "recraft_image_generator"
    description: str = "Generates an ad bannerimage using Recraft-AI and saves it locally."
    args_schema: Type[BaseModel] = RecraftGenInput
    return_direct: bool = False
    llm: OpenAI = None
    id: int = 1
    folder: str = "../output"
    config: dict = None
    agent: str = ""
    agent_turn: int = 1  # Add agent turn counter
    team_id: int = Field(default=0, description="Team ID for naming")  # 🎯 ADD: Team ID field
    round_number: int = Field(default=1, description="Current round number for naming")  # 🎯 FIXED: Properly declare as Pydantic field
    banner_size: str = Field(default="1024x1024", description="Banner size configuration from config file")  # 🎯 FIXED: Set default value from config file

    def __init__(self, config, agent=""):
        super().__init__()
       
        self.id = 1
        self.folder = config["SETTING"]["output_folder"]
        self.config = config
        
        # 🎯 ENHANCED AGENT NAME PARSING: Extract team ID from agent name
        self.agent = agent.replace(" ", "_") if agent else "unknown_agent"  # Sanitize agent name
        self.team_id = 0  # Default team ID
        
        # Extract team ID from agent name if it contains "Team" pattern
        if "Team" in self.agent:
            import re
            team_match = re.search(r'Team(\d+)', self.agent)
            if team_match:
                self.team_id = int(team_match.group(1))
                print(f"🎯 Extracted Team ID {self.team_id} from agent name: {self.agent}")
        
        self.agent_turn = 1
        self.round_number = 1  # Add round number tracking
        
        # 🎯 BANNER SIZE CONFIG: Read banner size from config file and set properly
        if hasattr(config, 'has_section') and config.has_section("BANNER_CONFIG") and config.has_option("BANNER_CONFIG", "banner_size"):
            self.banner_size = config.get("BANNER_CONFIG", "banner_size")
            print(f"📐 Banner size from config: {self.banner_size}")
        elif isinstance(config, dict) and "BANNER_CONFIG" in config and "banner_size" in config["BANNER_CONFIG"]:
            self.banner_size = config["BANNER_CONFIG"]["banner_size"]
            print(f"📐 Banner size from dict config: {self.banner_size}")
        else:
            self.banner_size = "1024x1024"  # 🎯 FIXED: Set default value from config file
            print(f"⚠️ No banner size in config, using default: {self.banner_size}")
        
        if self.config["IMAGE_GENERATION"]["FIRST_TURN_MODEL"] == "recraft":
            self.llm = OpenAI(
            base_url='https://external.api.recraft.ai/v1',
            api_key=config["KEYS"]["RECAFT_API_KEY"],
        )
        elif self.config["IMAGE_GENERATION"]["FIRST_TURN_MODEL"] == "gpt":
            if self.config["IMAGE_GENERATION"]["WHERE_4O"] == "openai":
                self.llm = OpenAI(api_key=config["KEYS"]["GPT_IMAGE_1_API_KEY"])
            elif self.config["IMAGE_GENERATION"]["WHERE_4O"] == "azure":
                self.llm = AzureOpenAI(api_key=config["KEYS"]["AZURE_OPENAI_API_IMAGE_KEY"], api_version="2025-04-01-preview", azure_endpoint=config["KEYS"]["AZURE_OPENAI_API_IMAGE_ENDPOINT"])

    def update_round_number(self, new_round_number):
        """
        Force update the round number for this tool instance
        """
        self.round_number = new_round_number
        print(f"🔧 [RecraftTool] Round number updated to {new_round_number} for agent {self.agent}")

    def _count_team_revisions_for_round(self, round_number):
        """
        Count actual revisions by examining generated files for a specific round
        """
        import glob
        
        pattern = f"{self.folder}/generated_image_*_Team{self.team_id}_Round{round_number}_*.png"
        files = glob.glob(pattern)
        
        max_revision = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
            import re
            match = re.search(r'_Team\d+_Round\d+_(\d+)\.png$', filename)
            if match:
                revision_num = int(match.group(1))
                max_revision = max(max_revision, revision_num)
        
        return max_revision

    def _run(
        self,
        prompt: str,
        size: str = None,  # 🎯 FIXED: Remove hardcoded default value, use configured size
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        
        # 🎯 DYNAMIC ROUND NUMBER: Get round_number from config/run_manager instead of instance attribute
        current_round = getattr(self, 'round_number', 1)  # Default fallback
        
        # Priority 1: Get from run_manager config (passed from team execution)
        if run_manager and hasattr(run_manager, 'metadata') and run_manager.metadata:
            if 'round_number' in run_manager.metadata:
                current_round = run_manager.metadata['round_number']
                print(f"🎯 [RecraftTool] Got round_number={current_round} from run_manager.metadata")
        
        # Priority 2: Get from run_manager config
        elif run_manager and hasattr(run_manager, 'config') and run_manager.config:
            if 'round_number' in run_manager.config:
                current_round = run_manager.config['round_number']
                print(f"🎯 [RecraftTool] Got round_number={current_round} from run_manager.config")
        
        # Priority 3: Check if there's a global config with round_number
        elif hasattr(self, 'config') and self.config:
            try:
                if hasattr(self.config, 'get'):
                    current_round = int(self.config.get("LLM", "current_round", fallback=current_round))
                elif isinstance(self.config, dict) and "current_round" in self.config:
                    current_round = int(self.config["current_round"])
                elif isinstance(self.config, dict) and "round_number" in self.config:
                    current_round = int(self.config["round_number"])
                print(f"🎯 [RecraftTool] Got round_number={current_round} from tool config")
            except:
                pass  # Use default if config parsing fails
        
        print(f"🔧 [RecraftTool] Using round_number={current_round} for Team {self.team_id}")
        
        # 🔒 HARD LIMIT ENFORCEMENT: Check revision limits before generating
        if hasattr(self, 'team_id'):
            current_revisions = self._count_team_revisions_for_round(current_round)
            max_revisions = getattr(self, 'max_revisions_per_round', 3)  # Default to 3 if not set
            
            # Try to get max_revisions from config
            if hasattr(self, 'config') and self.config:
                try:
                    if hasattr(self.config, 'get'):
                        max_revisions = int(self.config.get("LLM", "max_revisions_per_team", fallback=3))
                    elif isinstance(self.config, dict) and "LLM" in self.config:
                        max_revisions = int(self.config["LLM"].get("max_revisions_per_team", 3))
                except:
                    max_revisions = 3
            
            print(f"🔍 [RecraftTool] Team {self.team_id}, Round {current_round}: {current_revisions}/{max_revisions} revisions")
            
            if current_revisions >= max_revisions:
                error_msg = f"🚫 HARD LIMIT REACHED: Team {self.team_id} has already generated {current_revisions} revisions in Round {current_round} (limit: {max_revisions}). Image generation blocked."
                print(error_msg)
                raise RuntimeError(error_msg)
        
        # 🎯 BANNER SIZE ENFORCEMENT: Use config banner size if not specified
        if size is None:
            size = self.banner_size
            print(f"📐 Using configured banner size: {size}")
        else:
            print(f"📐 Using provided size: {size}")
        
        try:
            # 🎯 FIXED: Calculate correct revision number for current round (starts from 1 for each round)
            current_revisions = self._count_team_revisions_for_round(current_round)
            next_revision_number = current_revisions + 1  # Next revision in this round
            print(f"🔢 [RecraftTool] Team {self.team_id}, Round {current_round}: Generating revision {next_revision_number}")
            
            # Generate the image
            print("Generating image...")
            if self.config["IMAGE_GENERATION"]["FIRST_TURN_MODEL"] == "recraft":
                print(f"--- Running Recraft ImageGenTool ---")
                result = self.llm.images.generate(
                    prompt=prompt,
                    size=size,
                )
                image_url = result.data[0].url
                
                # 🎯 FIXED NAMING: Use calculated revision number instead of agent_turn
                # Format: generated_image_ImageResearcher_Team{TeamID}_Round{RoundNumber}_{RevisionNumber}.png
                agent_role = "ImageResearcher" if "ImageResearcher" in self.agent else "Agent"
                filepath = f"{self.folder}/generated_image_{agent_role}_Team{self.team_id}_Round{current_round}_{next_revision_number}.png"
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                image_bytes = requests.get(image_url).content
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                print(f"🎨 Image saved at {filepath}")
                
            elif self.config["IMAGE_GENERATION"]["FIRST_TURN_MODEL"] == "gpt":
                print(f"--- Running GPT ImageGenTool ---")
                result = self.llm.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size=size  # 🎯 FIXED: Use configured size instead of hardcoded value
                )
                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)
                
                # 🎯 FIXED NAMING: Use calculated revision number instead of agent_turn
                # Format: generated_image_ImageResearcher_Team{TeamID}_Round{RoundNumber}_{RevisionNumber}.png
                agent_role = "ImageResearcher" if "ImageResearcher" in self.agent else "Agent"
                filepath = f"{self.folder}/generated_image_{agent_role}_Team{self.team_id}_Round{current_round}_{next_revision_number}.png"
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                print(f"🎨 Image saved at {filepath}")

            # Summary
            text_summary = (
                f"✅ Image generated and saved at: `{filepath}`\n"
                f"📷 Prompt used:\n```text\n{prompt}\n```\n"
                f"🎯 Team {self.team_id}, Round {current_round}, Revision {next_revision_number}"
            )

            return {
                "text_summary": text_summary,
                "image_path": filepath,
                "recent_image_path": filepath  # Add this for state tracking
            }

        except Exception as e:
            print(f"❌ Error: {repr(e)}", file=sys.stderr)
            return {
                "text_summary": f"❌ Failed to generate image. Error: {repr(e)}",
                "image_path": None
            }



from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

import PIL.Image

class GeminiTool(BaseModel):
    prompt: str = Field(description="Instructions for the image generation.  \
                        When editing images, give instructions in the form of: Modify the text of 'Before edit' to 'After edit' or Move the object of 'Before edit' to 'After edit'. \
                        Make sure to put it into a numbered list. Furthermore, make sure to give explanations regarding the image(s) you are giving as input.")
    size: Literal["1024x1024"] = "1024x1024"  # 🎯 FIXED: Only allow configured banner size to prevent agent from choosing different sizes
    input_filepath: str = Field(description="Input a list of filepaths in string format to reference images. Format: [\"filepath1\", \"filepath2\", ...]")
    # output_filepath: str = Field(description="Output as a filepath to save the generated image.")
    @field_validator('input_filepath', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class GeminiImageGenTool(BaseTool):
    name: str = "gemini_image_generator"
    description: str = "Generates a new image according to the prompt and the image(s) provided.  \
        The input is a string of a list of filepaths to the images you can use in the format: [\"filepath1\", \"filepath2\", ...], and the prompt is the instructions for the image generation,  \
        explicitly containing explanations regarding the input image(s)."
    args_schema: Type[BaseModel] = GeminiTool
    return_direct: bool = False
    llm: Union[genai.Client, OpenAI] = None
    id: int = 1
    folder: str = "../output"
    disable_parallel_tool_use: bool = True
    config: dict = None
    running_tool: bool = False
    agent: str = ""
    team_id: int = Field(default=0, description="Team ID for naming")  # 🎯 ADD: Team ID field
    revision_count: int = 1  # Add revision counter
    round_number: int = Field(default=1, description="Current round number for naming")  # 🎯 FIXED: Properly declare as Pydantic field
    banner_size: str = Field(default="1024x1024", description="Banner size configuration from config file")  # 🎯 FIXED: Set default value from config file

    def __init__(self, config, agent=""):
        super().__init__()
        if config ["IMAGE_GENERATION"]["MODEL"] == "gpt":
           
            if config["IMAGE_GENERATION"]["WHERE_4O"] == "openai":
                from openai import OpenAI
                self.llm =OpenAI(api_key=config["KEYS"]["GPT_IMAGE_1_API_KEY"])
            elif config["IMAGE_GENERATION"]["WHERE_4O"] == "azure":
                from openai import AzureOpenAI
                self.llm =AzureOpenAI(api_key=config["KEYS"]["AZURE_OPENAI_API_IMAGE_KEY"], api_version="2025-04-01-preview", base_url=config["KEYS"]["GPT_IMAGE_1_BASE_URL"],)

        elif config ["IMAGE_GENERATION"]["MODEL"] == "gemini":
            self.llm = genai.Client(api_key=config["KEYS"]["GEMINI_API_KEY"])
        else:
            raise ValueError("Invalid model selected. Please choose 'gpt' or 'gemini'.")    
        
        self.id = 1
        self.folder = config["SETTING"]["output_folder"]
        self.config = config
        
        # 🎯 ENHANCED AGENT NAME PARSING: Extract team ID from agent name
        self.agent = agent.replace(" ", "_") if agent else "GraphicRevisor"  # Default to GraphicRevisor if no agent specified
        self.team_id = 0  # Default team ID
        
        # Extract team ID from agent name if it contains "Team" pattern
        if "Team" in self.agent:
            import re
            team_match = re.search(r'Team(\d+)', self.agent)
            if team_match:
                self.team_id = int(team_match.group(1))
                print(f"🎯 Extracted Team ID {self.team_id} from agent name: {self.agent}")
        
        self.revision_count = 1
        self.round_number = 1  # Add round number tracking

        # 🎯 BANNER SIZE CONFIG: Read banner size from config file and set properly
        if hasattr(config, 'has_section') and config.has_section("BANNER_CONFIG") and config.has_option("BANNER_CONFIG", "banner_size"):
            self.banner_size = config.get("BANNER_CONFIG", "banner_size")
            print(f"📐 Banner size from config: {self.banner_size}")
        elif isinstance(config, dict) and "BANNER_CONFIG" in config and "banner_size" in config["BANNER_CONFIG"]:
            self.banner_size = config["BANNER_CONFIG"]["banner_size"]
            print(f"📐 Banner size from dict config: {self.banner_size}")
        else:
            self.banner_size = "1024x1024"  # 🎯 FIXED: Set default value from config file
            print(f"⚠️ No banner size in config, using default: {self.banner_size}")

    def update_round_number(self, new_round_number):
        """
        Force update the round number for this tool instance
        """
        self.round_number = new_round_number
        print(f"🔧 [GeminiTool] Round number updated to {new_round_number} for agent {self.agent}")

    def _count_team_revisions_for_round(self, round_number):
        """
        Count actual revisions by examining generated files for a specific round
        """
        import glob
        
        pattern = f"{self.folder}/generated_image_*_Team{self.team_id}_Round{round_number}_*.png"
        files = glob.glob(pattern)
        
        max_revision = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
            import re
            match = re.search(r'_Team\d+_Round\d+_(\d+)\.png$', filename)
            if match:
                revision_num = int(match.group(1))
                max_revision = max(max_revision, revision_num)
        
        return max_revision

    def _count_team_edits_for_round(self, round_number):
        """
        Count actual edits by examining generated files for a specific round
        """
        import glob
        
        pattern = f"{self.folder}/generated_image_*_Team{self.team_id}_Round{round_number}_*.png"
        files = glob.glob(pattern)
        
        max_edit = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
            import re
            match = re.search(r'_Team\d+_Round\d+_(\d+)\.png$', filename)
            if match:
                edit_num = int(match.group(1))
                max_edit = max(max_edit, edit_num)
        
        return max_edit

    def _run(
        self,
        prompt: str,
        input_filepath: str = "",
        size: str = None,  # 🎯 FIXED: Remove hardcoded default value, use configured size
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        
        # 🎯 DYNAMIC ROUND NUMBER: Get round_number from config/run_manager instead of instance attribute
        current_round = getattr(self, 'round_number', 1)  # Default fallback
        
        # Priority 1: Get from run_manager metadata (passed from team execution)
        if run_manager and hasattr(run_manager, 'metadata') and run_manager.metadata:
            if 'round_number' in run_manager.metadata:
                current_round = run_manager.metadata['round_number']
                print(f"🎯 [GeminiTool] Got round_number={current_round} from run_manager.metadata")
        
        # Priority 2: Get from run_manager config
        elif run_manager and hasattr(run_manager, 'config') and run_manager.config:
            if 'round_number' in run_manager.config:
                current_round = run_manager.config['round_number']
                print(f"🎯 [GeminiTool] Got round_number={current_round} from run_manager.config")
        
        # Priority 3: Check if there's a global config with round_number
        elif hasattr(self, 'config') and self.config:
            try:
                if hasattr(self.config, 'get'):
                    current_round = int(self.config.get("LLM", "current_round", fallback=current_round))
                elif isinstance(self.config, dict) and "current_round" in self.config:
                    current_round = int(self.config["current_round"])
                elif isinstance(self.config, dict) and "round_number" in self.config:
                    current_round = int(self.config["round_number"])
                print(f"🎯 [GeminiTool] Got round_number={current_round} from tool config")
            except:
                pass  # Use default if config parsing fails
        
        print(f"🔧 [GeminiTool] Image editing (NOT counted as revision) - Round {current_round}, Team {self.team_id}")
        
        # 🔧 IMAGE EDIT LIMIT ENFORCEMENT: Check edit limits (separate from revision limits)
        if hasattr(self, 'team_id'):
            current_edits = self._count_team_edits_for_round(current_round)
            max_edits = 0  # Default to 0 (no edits allowed)
            
            # Try to get max_image_edits from config
            if hasattr(self, 'config') and self.config:
                try:
                    if hasattr(self.config, 'get'):
                        max_edits = int(self.config.get("LLM", "max_image_edits", fallback=0))
                    elif isinstance(self.config, dict) and "LLM" in self.config:
                        max_edits = int(self.config["LLM"].get("max_image_edits", 0))
                except:
                    max_edits = 0
            
            print(f"🔍 [GeminiTool] Team {self.team_id}, Round {current_round}: {current_edits}/{max_edits} edits")
            
            if current_edits >= max_edits:
                error_msg = f"🚫 EDIT LIMIT REACHED: Team {self.team_id} has already performed {current_edits} edits in Round {current_round} (limit: {max_edits}). Image editing blocked."
                print(error_msg)
                return {
                    "text_summary": error_msg,
                    "image_path": None
                }
        
        # 🔧 REMOVED: Hard limit enforcement for editing tools
        # Editing existing images does NOT count as revision, so no limits apply
        # Only RecraftImageGenTool (new image generation) has revision limits
        
        # 🎯 BANNER SIZE ENFORCEMENT: Use config banner size if not specified
        if size is None:
            size = self.banner_size
            print(f"📐 Using configured banner size: {size}")
        else:
            print(f"📐 Using provided size (GeminiTool): {size}")
        
        if self.running_tool:
            return {
                "text_summary": "⚠️ Tool is already running. Please wait for it to finish. Note that async tool is not implemented.",
                "image_path": None
            }
        if input_filepath is None or input_filepath == "":
            return {
                "text_summary": "❌ Input filepath is incorrect. You must input a list of filepaths to images.",
                "image_path": None
            }
        if not (input_filepath.startswith("[") and input_filepath.endswith("]")):
            input_filepath = [input_filepath]
        else:
            try:
                input_filepath = ast.literal_eval(input_filepath)
            except:
                return {
                    "text_summary": "❌ Input filepath is incorrect. You must input a string of list of filepaths to images.",
                    "image_path": None
                }

        # Smart file path resolution - resolve variable names to actual paths
        resolved_filepaths = []
        for filepath in input_filepath:
            # Extended list of variable names that should be resolved to actual paths
            if filepath in ['most_recent_image_filepath', 'recent_image_path', 'latest_image', 
                           'most_recent_image.png', 'latest_image.png', 'recent_image.png',
                           'most_recent_image', 'latest_image_path']:
                # Try to find the most recent image from common output directories
                output_base = self.folder
                
                # Search for recent images in agent subdirectories
                recent_image = None
                max_timestamp = 0
                
                # First try to find images for the current agent/team
                current_agent_pattern = self.agent.replace("_", "").lower()
                
                if os.path.exists(output_base):
                    # Priority 1: Find images specifically for this agent/team
                    agent_specific_images = []
                    other_images = []
                    
                    for filename in os.listdir(output_base):
                        if filename.startswith('generated_image_') and filename.endswith('.png'):
                            try:
                                parts = filename.replace('.png', '').split('_')
                                if len(parts) >= 4:  # generated_image_agentname_turn
                                    agent_name = '_'.join(parts[2:-1]).lower()
                                    turn = int(parts[-1])
                                    # Check if this image belongs to the current agent's team
                                    if current_agent_pattern in agent_name or agent_name in current_agent_pattern:
                                        agent_specific_images.append((filename, turn, os.path.join(output_base, filename)))
                                    else:
                                        other_images.append((filename, turn, os.path.join(output_base, filename)))
                            except ValueError:
                                continue
                    
                    # Find the most recent image for this specific agent
                    if agent_specific_images:
                        agent_specific_images.sort(key=lambda x: x[1], reverse=True)  # Sort by turn number
                        recent_image = agent_specific_images[0][2]  # Get file path
                        print(f"🎯 Found agent-specific image: {recent_image}")
                    elif other_images:
                        # Fallback to any recent image if no agent-specific images found
                        other_images.sort(key=lambda x: x[1], reverse=True)
                        recent_image = other_images[0][2]
                        print(f"🔄 Using fallback image: {recent_image}")
                
                if recent_image and os.path.exists(recent_image):
                    resolved_filepaths.append(recent_image)
                    print(f"🔍 Resolved '{filepath}' to: {recent_image}")
                else:
                    return {
                        "text_summary": f"❌ Failed to resolve file path variable '{filepath}'. No recent images found in output directories.",
                        "image_path": None
                    }
            else:
                # Check if it's an absolute or relative path that exists
                if os.path.exists(filepath):
                    resolved_filepaths.append(filepath)
                else:
                    # Try to resolve it relative to the output folder
                    potential_path = os.path.join(self.folder, filepath)
                    if os.path.exists(potential_path):
                        resolved_filepaths.append(potential_path)
                        print(f"🔍 Resolved relative path '{filepath}' to: {potential_path}")
                    else:
                        return {
                            "text_summary": f"❌ Failed to find image file: {filepath}. Please ensure the file path is correct and the file exists. Available in {self.folder}: {os.listdir(self.folder) if os.path.exists(self.folder) else 'Directory not found'}",
                            "image_path": None
                        }
        
        input_filepath = resolved_filepaths

        self.running_tool = True
        try:
            # Generate the image
            text_input = prompt
            
            if self.config["IMAGE_GENERATION"]["MODEL"] == "gemini":
                if input_filepath is not None and input_filepath != "":
                    image_list = []
                    for filepath in input_filepath:
                        # Validate file exists before trying to open
                        if not os.path.exists(filepath):
                            return {
                                "text_summary": f"❌ Failed to generate image. Input file not found: {filepath}. Please ensure the file path is correct and the file exists.",
                                "image_path": None
                            }
                        image = PIL.Image.open(filepath)
                        image_list.append(image)
                    contents = [text_input, image_list]
                else:
                    image = None
                    contents = [text_input]
                print(f"--- Running GenAI Gemini Tool ---")
                response = self.llm.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=contents,
                    config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                    ),
                )
            elif self.config["IMAGE_GENERATION"]["MODEL"] == "gpt":
                print(f"--- Running OpenAI Image Edit Tool ---") 
                print(input_filepath)

                # Validate files exist before trying to open them
                for filepath in input_filepath:
                    if not os.path.exists(filepath):
                        return {
                            "text_summary": f"❌ Failed to generate image. Input file not found: {filepath}. Please ensure the file path is correct and the file exists.",
                            "image_path": None
                        }
                result = self.llm.images.edit(
                model="gpt-image-1",
                image=[
                    open(filepath, "rb") for filepath in input_filepath
                ],
                prompt=text_input,
                size=size  # 🎯 FIXED: Use configured size instead of hardcoded value
            )
            
            # For revision naming, find original agent name and team ID from input file
            original_agent_role = "GraphicRevisor"
            original_team_id = self.team_id
            original_turn = 1
            
            if resolved_filepaths and len(resolved_filepaths) > 0:
                input_filename = os.path.basename(resolved_filepaths[0])
                if input_filename.startswith('generated_image_'):
                    try:
                        # Parse: generated_image_ImageResearcher_Team1_Round1_3.png
                        parts = input_filename.replace('.png', '').split('_')
                        if len(parts) >= 6:  # generated_image_ImageResearcher_Team1_Round1_3
                            original_agent_role = parts[2]  # ImageResearcher
                            # Extract team ID
                            team_part = parts[3]  # Team1
                            if team_part.startswith('Team'):
                                original_team_id = int(team_part[4:])  # Extract number after 'Team'
                            # Extract round and turn info
                            round_part = parts[4]  # Round1
                            if round_part.startswith('Round'):
                                # Keep the same round for revisions, or use current round_number
                                pass
                            original_turn = int(parts[5])  # 3
                    except Exception as parse_error:
                        print(f"⚠️ Could not parse input filename: {input_filename}, error: {parse_error}")
                        # Use defaults
            
            # 🎯 FIXED: Calculate correct revision number for current round (starts from 1 for each round)
            # For image editing, we create new images but with sequential naming within the round
            current_revisions = self._count_team_revisions_for_round(current_round)
            next_revision_number = current_revisions + 1  # Next revision in this round
            print(f"🔢 [GeminiTool] Team {original_team_id}, Round {current_round}: Generating edit revision {next_revision_number}")
            
            # 🎯 FIXED NAMING: Use calculated revision number instead of revision_count
            # Format: generated_image_ImageResearcher_Team{TeamID}_Round{RoundNumber}_{RevisionNumber}.png
            output_filepath = f"{self.folder}/generated_image_{original_agent_role}_Team{original_team_id}_Round{current_round}_{next_revision_number}.png"
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            
            buffered = BytesIO()
            if self.config["IMAGE_GENERATION"]["MODEL"] == "gemini":
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        image.save(output_filepath)
                    else:
                        return {
                            "text_summary": "❌ No image generated. Please try again.",
                            "image_path": None
                        }
                image.save(buffered, format="PNG")
            elif self.config["IMAGE_GENERATION"]["MODEL"] == "gpt":
                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)
                with open(output_filepath, "wb") as f:
                    f.write(image_bytes)
                buffered = BytesIO(image_bytes)

            # Summary
            text_summary = (
                f"✅ Image edited and saved at: `{output_filepath}`\n"
                f"📷 Edit prompt used:\n```text\n{prompt}\n```\n"
                f"🎯 Team {original_team_id}, Round {current_round}, Edit Revision {next_revision_number}"
            )

            # 🔧 REMOVED: No longer incrementing revision_count since we calculate revision number dynamically
            # self.revision_count += 1

            # Encode generated image into base64, and format it as a data URI
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{encoded_image}"

            image_return = {
                "type": "image_url",
                "image_url": {
                    "url": data_uri
                }
            }
            
            self.running_tool = False
            return {
                "text_summary": text_summary,
                "image_path": output_filepath,
                "image_return": image_return,
                "recent_image_path": output_filepath  # Add this for state tracking
            }

        except Exception as e:
            self.running_tool = False
            import traceback
            traceback.print_exc()
            print(f"❌ Error: {repr(e)}", file=sys.stderr)
            return {
                "text_summary": f"❌ Failed to generate image. Error: {repr(e)}",
                "image_path": None
            }

    async def _arun(
        self,
        prompt: str,
        input_filepath: str = "",
        size: str = None,  # 🎯 FIXED: Remove hardcoded default value, use configured size
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        # 🎯 BANNER SIZE ENFORCEMENT: Use config banner size if not specified
        if size is None:
            size = self.banner_size
            print(f"📐 [Async] Using configured banner size: {size}")
        
        return {
            "text_summary": "⚠️ Async tool not implemented.",
            "image_path": None
        }



from sympy import sympify, S, symbols,  Not, Or, And, Implies, Equivalent
import itertools


class MultStr(BaseModel):
    in_str: str = Field(description="Input as a list of strings, in the form of: [Expression1, Expression2, ...]")



class TruthTableGenerator(BaseTool):
    name: str = "truthtable_generator"
    description: str = "Returns the truth table for a given list of Boolean expression. Always use SymPy-style logical operators: \
                    And(A, B) for AND, Or(A, B) for OR, Not(A) for NOT, Implies(A, B) for IMPLIES, and Equivalent(A, B) for BICONDITIONAL. \
                    Parentheses can be used for grouping. Example: ['And(Or(Not(A), B), Implies(C, A))', ...]."

    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if in_str[0] != "[":
            in_str = [in_str]
        else:
            in_str = ast.literal_eval(in_str)
        # print(in_str)
        try:
            expected_symbols = [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ]
            local_dict = {name: symbols(name) for name in expected_symbols}
            local_dict.update({
                "~": Not,  # Negation
                "&": And,  # Logical AND
                "|": Or,   # Logical OR
                ">>": Implies,  # Logical implication
                "EQ": Equivalent  # Logical equivalence
            })

            # Parse the expressions
            parsed_expressions = [sympify(expr, locals=local_dict) for expr in in_str]
            if len(in_str) > 1:
                parsed_expressions.append(Equivalent(*parsed_expressions))
                in_str.append("Same Value for All Formulas")
            
            # Extract free symbols from all expressions
            free_syms = set().union(*(expr.free_symbols for expr in parsed_expressions))
            variables = sorted({str(s) for s in free_syms})

            # If there are no variables, each expression is constant
            if not variables:
                results = ["T" if expr else "F" for expr in parsed_expressions]
                return "Constant expressions:\n" + "\n".join(f"{expr} = {res}" for expr, res in zip(in_str, results))

            # Generate all possible truth assignments for the variables.
            truth_combinations = list(itertools.product([False, True], repeat=len(variables)))

            # Build the truth table rows.
            headers = " | ".join(variables) + " | " + " | ".join(in_str)
            separator = "-" * len(headers)
            table_rows = [headers, separator]

            same_val_list = []

            for values in truth_combinations:
                # Map variable names to boolean values.
                var_dict = dict(zip(variables, values))
                # Evaluate all expressions with these values.
                results = ["T" if expr.subs(var_dict) else "F" for expr in parsed_expressions]
                row_values = " | ".join("T" if v else "F" for v in values)
                table_rows.append(f"{row_values} | " + " | ".join(results))
                same_val_list.append(results[-1] == "T")
            
            if sum(same_val_list) == len(same_val_list):
                table_rows.append("\nIMPORTANT: THE GIVEN PROPOSITIONS ARE: **Logically Equivalent**")
                print("Logically Equivalent")
            elif sum(same_val_list) == 0:
                table_rows.append("\nIMPORTANT: THE GIVEN PROPOSITIONS ARE: **Contradictory**")
                print("Contradictory")


            return "\n".join(table_rows)

        except Exception as e:
            return f"Error processing expressions: {str(e)}"



class CounterexampleVerifier(BaseTool):
    name: str = "counterexample_verifier"
    description: str = "Verifies whether a given set of truth values serves as a counterexample to an argument. "\
                        "Input consists of premises, a conclusion, and a dictionary specifying truth values for variables, in the form of: "\
                        "{{\"premises\": [Premis1, ...], \"conclusion\": Conclusion, \"truth_values\": [{{variable1: \"True/False\", ...}}, ...]}}"\
                        "Uses SymPy-style logical operators: And(A, B), Or(A, B), Not(A), Implies(A, B), Equivalent(A, B). Make sure to give True False as a string."

    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            expected_symbols = [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ]
            local_dict = {name: symbols(name) for name in expected_symbols}
            local_dict.update({
                "~": Not,  # Negation
                "&": And,  # Logical AND
                "|": Or,   # Logical OR
                ">>": Implies,  # Logical implication
                "EQ": Equivalent  # Logical equivalence
            })
            in_dict = ast.literal_eval(in_str)
            premises = in_dict["premises"]
            if len(premises) == 0:
                return "Empty premis, no evaluation possible."
            conclusion = in_dict["conclusion"]

            result = ""

            print(in_dict["truth_values"])

            for truth_value_list in in_dict["truth_values"]:
                truth_values = truth_value_list
                for key in truth_values.keys():
                    if truth_values[key] in ["True", "true", "T"]:
                        truth_values[key] = True
                    else:
                        truth_values[key] = False

                # Parse the premises and conclusion
                parsed_premises = [sympify(expr, locals=local_dict) for expr in premises]
                parsed_conclusion = sympify(conclusion, locals=local_dict)

                # Ensure the provided truth values match expected variables
                all_variables = set().union(*(expr.free_symbols for expr in parsed_premises + [parsed_conclusion]))
                var_dict = {str(var): truth_values[str(var)] for var in all_variables if str(var) in truth_values}

                # Evaluate premises and conclusion under the given truth assignment
                premises_results = [expr.subs(var_dict) for expr in parsed_premises]
                conclusion_result = parsed_conclusion.subs(var_dict)

                # Convert results to Boolean values
                premises_truths = [bool(result) for result in premises_results]
                conclusion_truth = bool(conclusion_result)

                # A counterexample occurs when all premises are true and the conclusion is false
                if all(premises_truths) and not conclusion_truth:
                    result += "For " + str(truth_value_list) + ": Valid counterexample. The given truth values make all premises true and the conclusion false.\n"
                else:
                    result += "For " + str(truth_value_list) + ": Not a counterexample.The given truth values do not satisfy the conditions for a counterexample.\n"
            return result
        except Exception as e:
            return f"Error processing expressions: {str(e)}"


import csv
class RejectWordTool(BaseTool):
    name: str = "reject_words"
    description: str = "A reject word checker. Checks whether each sentence contains words that should not be included. Takes as input a list composed in the form: [sentence1, sentence2, ...]."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False
    reject_list: list = []

    def __init__(self, file_path):
        super().__init__()
        self.reject_list = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.reject_list += row


    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        return_str = ""
        for sent in in_sent:
            rejected = []
            for reject in self.reject_list:
                if reject in sent:
                    rejected.append(reject)
            
            if len(rejected) > 0:
                return_str += f"{sent}: Rejected {rejected}\n"
            else:
                return_str += f"{sent}: Good\n"

        return return_str





class FileWriterInput(BaseModel):
    file_name: str = Field(description="The name of the file in question.")
    content: str = Field(description="The content to be written into the file. Put 'None' if you want to read from the file.")

# A tool for writing outputs into a file. The inputs are file name and the content to be written.
class FileReadWriterTool(BaseTool):
    name: str = "file_writer"
    description: str = "A tool for writing outputs into a file, and reading from a file. The inputs are file name and the content to be written. Put 'None' if you want to read from the file."
    args_schema: Type[BaseModel] = FileWriterInput
    return_direct: bool = False
    folder: str = ""

    def __init__(self, config):
        super().__init__()
        self.folder = config["SETTING"]["output_folder"]

    def _run(
        self, file_name: str, content: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if content == "None":
            with open(file_name, "r") as file:
                return file.read()
        else:
            file_path = f"{self.folder}/{file_name}"
            with open(file_path, "w") as file:
                file.write(content)
        return f"File {file_path} written successfully."




def retrieveSearches():
    if not os.path.exists('../outputs/searches.pkl'):
        return {}
    else:
        with open('../outputs/searches.pkl', 'rb') as f:
            return pickle.load(f)

def saveSearches(key_dict):
    print("\nNote: Saved to backup")
    with open('../outputs/searches.pkl', 'wb') as f:
        pickle.dump(key_dict, f)



def getSerpTool():
    # Comment out if live SerpAPI is needed
    return SerpAPITool()

    search_tool = load_tools(["serpapi"])
    search_tool[0].name = "google_search"
    return search_tool[0]



from langchain_core.tools import Tool


def getTools(sel_tools, config, llm, agent_name=""):
    """ 
    0: SerpAPI
    1: Counter
    2,3: Ad Retriever
    4: Output
    5: Click Aggregator
    6: Python
    """
    if isinstance(sel_tools, str):
        sel_tools = ast.literal_eval(sel_tools)
    agent_tools = []
    
    # 🎯 ROUND NUMBER SUPPORT: Extract round_number from config if available
    current_round = 1  # Default
    if hasattr(config, 'get'):
        try:
            current_round = int(config.get('LLM', 'round_number', fallback=1))
        except:
            current_round = 1
    elif hasattr(config, '__getitem__'):
        try:
            current_round = config.get("round_number", 1)
        except:
            current_round = 1
    
    print(f"🔧 getTools: Using round_number={current_round} for agent {agent_name}")
    
    if 0 in sel_tools:
        os.environ["SERPAPI_API_KEY"] = config['KEYS']['SERPAPI_API_KEY']  
        agent_tools.append(getSerpTool())
    if 1 in sel_tools:
        agent_tools.append(CustomCounterTool())

    if 2 in sel_tools:
        KW_loader = CSVLoader(config["SETTING"]["initial_keyword_data"])
        KW_retriever = customized_trend_retriever(KW_loader, str(config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  \
                                                  str(config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

        agent_tools.append(create_retriever_tool(
            KW_retriever,
            str(config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
            str(config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
        ))

    if 3 in sel_tools:
        exampler_loader = TextLoader(str(config['SETTING']['rule_data']))
        exampler_retriever = customized_trend_retriever(exampler_loader, str(config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  \
                                                        str(config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 

        agent_tools.append(create_retriever_tool(
            exampler_retriever,
            str(config['TOOL']['RULE_RETRIEVAL_NAME']),
            #'Search',
            str(config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
        ))
    
    if 4 in sel_tools:
        agent_tools.append(OutputTool())

    if 5 in sel_tools:
        agent_tools.append(ClickAggregator(config["SETTING"]["initial_keyword_data"]))

    if 6 in sel_tools:
        python_repl = PythonREPL()
        agent_tools.append(Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        ))
    
    if 7 in sel_tools:
        agent_tools.append(RejectWordTool(config['TOOL']['REJECT_WORD_CSV']))
    
    if 8 in sel_tools:
        agent_tools.append(TruthTableGenerator())
    if 9 in sel_tools:
        agent_tools.append(CounterexampleVerifier())
    
    if 10 in sel_tools:
        agent_tools.append(ImageToBase64Tool())

    if 11 in sel_tools:
        recraft_tool = RecraftImageGenTool(config, agent=agent_name)
        recraft_tool.round_number = current_round  # Set initial round number
        agent_tools.append(recraft_tool)
        
    if 12 in sel_tools:
        gemini_tool = GeminiImageGenTool(config, agent=agent_name)
        gemini_tool.round_number = current_round  # Set initial round number
        agent_tools.append(gemini_tool)
        
    if 13 in sel_tools:
        recraft_tool = RecraftImageGenTool(config, agent=agent_name)
        recraft_tool.round_number = current_round  # Set initial round number
        agent_tools.append(recraft_tool)

    if 14 in sel_tools:
        agent_tools.append(FileReadWriterTool(config))
    
    return agent_tools


def update_tools_round_number(tools, new_round_number):
    """
    Helper function to update round numbers for all image generation tools
    This should be called when starting a new round
    """
    updated_count = 0
    for tool in tools:
        if hasattr(tool, 'update_round_number') and hasattr(tool, 'name'):
            tool_name = getattr(tool, 'name', 'UnknownTool')
            if 'image' in tool_name.lower() and 'generator' in tool_name.lower():
                tool.update_round_number(new_round_number)
                updated_count += 1
    
    if updated_count > 0:
        print(f"🔄 Updated {updated_count} image generation tools to Round {new_round_number}")
    
    return updated_count

