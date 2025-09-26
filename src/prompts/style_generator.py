#!/usr/bin/env python3
"""
Dynamic Style Generator
根据logo和prompt动态生成适合的design style候选
"""

import os
import json
import time
from PIL import Image
import base64
import io
from typing import List, Dict, Tuple, Optional
import pdb

class StyleAnalysisAgent:
    """Style analysis AI Agent"""
    
    def __init__(self, llm, config=None):
        self.llm = llm
        self.config = config
        
    def analyze_logo_and_generate_styles(self, logo_path: str, prompt_text: str, num_styles: int = 5) -> List[Dict]:
        """
        analyze logo and prompt, generate suitable style candidates
        
        Args:
            logo_path: logo file path
            prompt_text: product description text
            num_styles: number of styles to generate
            
        Returns:
            List of style dictionaries with style_name, philosophy
        """
        print(f"🎨 start analyzing logo and prompt to generate style candidates...")
        print(f"📁 Logo file: {logo_path}")
        print(f"📝 Prompt: {prompt_text}")
        
        # read logo file and convert to base64
        logo_base64 = self._encode_image_to_base64(logo_path)
        
        # build analysis prompt
        analysis_prompt = self._build_style_analysis_prompt(prompt_text, num_styles)
        
        # Try to use the LLM for style analysis (with image if possible)
        # Try vision-enabled LLM call with image
        from langchain.schema import HumanMessage
        
        message = HumanMessage(content=[
            {"type": "text", "text": analysis_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{logo_base64}"}}
        ])
        response = self.llm.invoke([message])
    

        # parse response and extract style
        styles = self._parse_style_response(response.content if hasattr(response, 'content') else str(response))
        
        print(f"✅ successfully generated {len(styles)} style candidates")
        return styles
            
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """encode image file to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"⚠️ cannot read image file: {e}")
            return None
    
    def _build_style_analysis_prompt(self, prompt_text: str, num_styles: int) -> str:
        """build system prompt for logo+prompt analysis"""
        return f"""You are a senior brand design expert. Please analyze the provided logo image and product description to generate {num_styles} different style candidates for banner design.

## Product Description
{prompt_text}

## Analysis Task
Please carefully analyze the visual elements of the logo:
1. **Color Scheme**: The main colors and color scheme used in the logo
2. **Design Style**: The design style of the logo 
3. **Font Characteristics**: The font style of the text in the logo
4. **Graphic Elements**: Icons, graphics, or decorative elements in the logo
5. **Overall Feel**: The brand tone and emotion conveyed by the logo

## Output Requirements
Generate {num_styles} different style candidates for banner design, each style must:
- Be consistent with the brand tone of the logo
- Be suitable for the characteristics of the target product
- Be visually harmonious with the logo
- Have unique design characteristics
- Include a rough imagination of the banner content that fits the style and differs from other styles
- We do not want the white background for the banner image so avoid such design.
- The philosophy should be generate specific design plan (for example,  specific ad text), just give me big, generalized design style or big design direction"

Please strictly follow the following JSON format:

```json
{{
  "styles": [
    {{
      "style_name": "Style Name ",
      "philosophy": "Design Philosophy",
       }}
  ]
}}
```

Ensure that each style is unique and practical, and can guide actual banner design work."""

    def _build_text_only_analysis_prompt(self, prompt_text: str, num_styles: int) -> str:
        """build text-only analysis prompt (when image cannot be processed)"""
        return f"""You are a senior brand design expert. Please generate {num_styles} different style candidates for banner design based on the following product description.

## Product Description
{prompt_text}

## Analysis Task
Please analyze the product description:
1. **Target Audience**: The main target user group of the product
2. **Brand Tone**: The brand tone that the product should convey (professional, fashionable, reliable, etc.)
3. **Industry Characteristics**: The design preferences of the product's industry
4. **Competitive Environment**: How to stand out in this field

## Output Requirements
Generate {num_styles} different style candidates for banner design, each style must:
- Be suitable for the target product and audience
- Have a clear design direction
- Be able to stand out in competition
- Cover different design strategies
- Each style should be different from the others and have its own unique design characteristics
- A rought imagination of the content of the banner which is suitable for the style and should be different from the others

Please strictly follow the following JSON format:

```json
{{
  "styles": [
    {{
      "style_name": "Style Name",
      "philosophy": "Design Philosophy",
    }}
  ]
}}
```

Ensure that each style is unique and practical, and can guide actual banner design work."""

    def _parse_style_response(self, response_text: str) -> List[Dict]:
        """parse LLM response and extract style information"""
        try:
            # try to extract JSON from response
            import re
            
            # find JSON code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # try to find raw JSON
                json_match = re.search(r'\{.*"styles".*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    raise ValueError("cannot find JSON formatted style data in response")
            
            # parse JSON
            data = json.loads(json_text)
            
            if "styles" in data and isinstance(data["styles"], list):
                return data["styles"]
            else:
                raise ValueError("JSON format is incorrect, missing styles array")
                
        except Exception as e:
            print(f"⚠️ Parsing style response failed: {e}")
            print(f"Original response: {response_text[:500]}...")
            return []
    
    def _generate_fallback_styles(self, prompt_text: str, num_styles: int) -> List[Dict]:
        """Generate fallback styles (when AI analysis fails)"""
        print("🔄 Using fallback style generation strategy...")
        
        # base style template
        base_styles = [
            {
                "style_name": "Professional and Authoritative",
                "philosophy": "Establish brand authority, convey reliable and professional image",

            },
            {
                "style_name": "Modern and Simple",
                "philosophy": "Less is more, highlight product essence",

            },
            {
                "style_name": "Vibrant and Stylish",
                "philosophy": "Attract young audience, convey innovation and energy",

            },
            {
                "style_name": "Elegant and Refined",
                "philosophy": "Highlight the high-end positioning and refined quality of the product",
            },
            {
                "style_name": "Warm and Friendly",
                "philosophy": "Bring users closer, build emotional connection",
            }
        ]
        
        # adjust style based on prompt content
        prompt_lower = prompt_text.lower()
        
        # adjust priority based on keywords
        if any(word in prompt_lower for word in ['tech', 'ai', 'digital', 'software', 'app']):
            # tech/ai product: prioritize modern and simple and professional and authoritative
            return base_styles[:num_styles]
        elif any(word in prompt_lower for word in ['beauty', 'fashion', 'lifestyle', 'premium']):
            # beauty/cosmetic product: prioritize elegant and refined and stylish and energetic
            return [base_styles[3], base_styles[2]] + base_styles[:3][:num_styles-2]
        elif any(word in prompt_lower for word in ['food', 'family', 'home', 'care']):
            # family/home product: prioritize warm and friendly
            return [base_styles[4]] + base_styles[:4][:num_styles-1]
        else:
            # default: return top num_styles
            return base_styles[:num_styles]


def select_best_styles(styles: List[Dict], logo_path: str, prompt_text: str, llm, target_count: int) -> Tuple[List[Dict], List[int]]:
    """
    select the best few styles from generated styles that are most suitable for logo and prompt
    
    Args:
        styles: generated all style list
        logo_path: logo file path
        prompt_text: product description
        llm: language model instance
        target_count: number of styles to select
        
    Returns:
        Tuple of (selected best style list, selected style indices (1-based for team IDs))
    """
    if len(styles) <= target_count:
        print(f"📊 Style number({len(styles)}) <= target number({target_count}), return all styles")
        style_indices = list(range(1, len(styles) + 1))  # 1-based indices
        return styles, style_indices
    
    print(f"🔍 select the best {target_count} styles from {len(styles)} styles...")
    
    # build style selection prompt
    selection_prompt = f"""You are a brand design consultant. Please analyze the product and select the {target_count} MOST SUITABLE design styles from the following {len(styles)} style candidates.

## Product Description
{prompt_text}

## Available Style Candidates
"""
    
    for i, style in enumerate(styles, 1):
        selection_prompt += f"""
### Style {i}: {style['style_name']}
- **Philosophy**: {style['philosophy']} 

"""
    
    selection_prompt += f"""

## Selection Criteria
Consider the following factors when selecting the {target_count} best styles:
1. **Brand Alignment**: How well does the style match the product's brand positioning?
2. **Target Audience Fit**: Does the style appeal to the product's target demographic?
3. **Industry Appropriateness**: Is the style suitable for this industry/product category?
4. **Visual Distinctiveness**: How distinctive is the style among all the styles?
5. **Practical Implementation**: Can this style be effectively executed in a banner format?

## Output Requirements
Please select exactly {target_count} styles and provide your reasoning. Output in the following JSON format:

```json
{{
  "selected_styles": [
    {{
      "style_number": 1,
      "style_name": "Style Name",
      "selection_reason": "Detailed reason why this style was selected",
      "ranking_score": 95,
      "strengths": ["strength1", "strength2", "strength3"]
    }}
  ],
  "selection_summary": "Overall reasoning for the selection strategy"
}}
```

Ensure the selected styles provide good variety while all being highly suitable for the product."""

    # use LLM to select style
    response = llm.invoke(selection_prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
  
    # parse selection result
    selected_indices = _parse_selection_response(response_text, len(styles))
    
    if selected_indices:
        selected_styles = [styles[i] for i in selected_indices]
        # Convert 0-based indices to 1-based team IDs (original style positions)
        team_ids = [i + 1 for i in selected_indices]
        print(f"✅ successfully selected {len(selected_styles)} best styles:")
        for i, (style, team_id) in enumerate(zip(selected_styles, team_ids), 1):
            print(f"   {i}. {style['style_name']} (original style #{team_id})")
        return selected_styles, team_ids
    else:
        # Fallback: select first target_count styles
        print(f"⚠️ Failed to parse style selection, using first {target_count} styles as fallback")
        selected_styles = styles[:target_count]
        team_ids = list(range(1, target_count + 1))
        return selected_styles, team_ids


def _parse_selection_response(response_text: str, total_styles: int) -> List[int]:
    """parse style selection response"""
    try:
        import re
        import json
        
        # find JSON code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # try to find raw JSON
            json_match = re.search(r'\{.*"selected_styles".*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                raise ValueError("cannot find JSON formatted selection result")
        
        # parse JSON
        data = json.loads(json_text)
        
        if "selected_styles" in data and isinstance(data["selected_styles"], list):
            # extract style index (convert to 0-based index)
            selected_indices = []
            for selected in data["selected_styles"]:
                if "style_number" in selected:
                    # convert to 0-based index
                    index = int(selected["style_number"]) - 1
                    if 0 <= index < total_styles:
                        selected_indices.append(index)
            
            return selected_indices
        else:
            raise ValueError("JSON format is incorrect, missing selected_styles array")
            
    except Exception as e:
        print(f"⚠️ parsing style selection response failed: {e}")
        return []


def generate_dynamic_styles(logo_path: str, prompt_text: str, llm, config=None, num_styles: int = 5, num_teams: int = None) -> Tuple[List[Dict], List[int]]:

    if num_teams is None:
        if config:
            try:
                if hasattr(config, 'get'):
                    num_teams = int(config.get("LLM", "num_teams", fallback=num_styles))
                elif isinstance(config, dict):
                    num_teams = config.get("LLM", {}).get("num_teams", num_styles)
                else:
                    num_teams = num_styles
            except:
                num_teams = num_styles
        else:
            num_teams = num_styles
    
    print(f"dynamic style generation config:")
    print(f"   candidate pool size: {num_styles} styles")
    print(f"   final selected number: {num_teams} styles")
    
    # step 1: generate style candidate pool
    analyzer = StyleAnalysisAgent(llm, config)
    all_styles = analyzer.analyze_logo_and_generate_styles(logo_path, prompt_text, num_styles)
    
    if not all_styles:
        print(" cannot generate style candidates")
        return [], []
    
    # step 2: select the best few from candidates
    if len(all_styles) > num_teams:
        selected_styles, selected_team_ids = select_best_styles(all_styles, logo_path, prompt_text, llm, num_teams)
    else:
        selected_styles = all_styles
        selected_team_ids = list(range(1, len(all_styles) + 1))  # 1-based indices
        print(f"📊 generated styles number({len(all_styles)}) <= target number({num_teams}), use all styles")
    
    return selected_styles, selected_team_ids


def format_styles_for_competitive_template(styles: List[Dict], banner_size: str = "1024x1024", selected_team_ids: List[int] = None) -> Dict:
    """
    format dynamic generated styles to competitive template format
    
    Args:
        styles: dynamic generated style list
        banner_size: banner size
        selected_team_ids: selected team ID list, used to correctly map
        
    Returns:
        Dictionary containing formatted team configurations
    """
    formatted_teams = {}
    
    # if selected_team_ids is not provided, use default sequence
    if selected_team_ids is None:
        selected_team_ids = list(range(1, len(styles) + 1))
    
    for i, style in enumerate(styles):
        # use actual selected team ID, not consecutive numbering
        team_id = selected_team_ids[i] if i < len(selected_team_ids) else i + 1
        team_key = f"BannerTeam{team_id}"
        
        # format style information to prompt
        style_prompt = f"Focus on {style['style_name']} design. "
        
        # Build style philosophy with banner content if available
        style_philosophy = f"Design Philosophy: {style['philosophy']}"
        
        # Add banner content suggestion if available
        if 'banner_content' in style and style['banner_content']:
            style_philosophy += f" Banner Content Suggestion: {style['banner_content']}."
        
        formatted_teams[team_key] = {
            "team": f"BannerGeneration{team_id}",
            "return": "FINISH" if i == len(styles) - 1 else "Default",
            "prompt": f"Generate Banner for {{item}}. {style_prompt}",
            "additional_prompt": style_philosophy,
            "ContentCreationTeam": _get_content_creation_team_config(f"BannerGeneration{team_id}", banner_size),
            "EvaluationTeam": _get_evaluation_team_config(f"BannerGeneration{team_id}"),
            "GraphicRevisor": _get_graphic_revisor_config()
        }
    
    print(f"🎯 formatted team config: {list(formatted_teams.keys())} (corresponding team ID: {selected_team_ids[:len(styles)]})")
    return formatted_teams


def _get_content_creation_team_config(team_name: str, banner_size: str) -> Dict:
    """get ContentCreationTeam config"""
    return {
        "team": "ContentCreation",
        "return": team_name,
        "prompt": "Create the text content and find high-quality images for the product we are advertising. You only need to generate 1 image and the text to go with it. Consider any judge feedback provided.",
        "additional_prompt": "Ensure the text and images are cohesive and high-quality. If judge feedback is available, prioritize addressing the mentioned weaknesses while maintaining strengths.",
        "Copywriter": {
            "prompt": "You are a Copywriter responsible for creating the text content for the AD banner, including the headline, subheadline, and CTA text. You can use tools to find information about the product. CTA text used in CTA button must be short catchy and concise. Do not generate untrue, misleading and incorrect information. If judge feedback is provided, incorporate suggestions for text improvements.",
            "tools": [4]
        },
        "ImageResearcher": {
            "prompt": f"You are an Image Researcher responsible for finding high-quality images of the product to be used in the banner. Follow the instructions given by the ContentCreation Supervisor. You need to pass the detailed information and detailed ad image generation request to the image generation tool. Only generate 1 image at a time. Only Generate 1 logo in the image. All text (including logo text, CTA text, text in the image, etc.) in the image need to be high contrast and visible. If judge feedback is provided, adjust the visual style accordingly. 🎯 IMPORTANT: Always use the banner size {banner_size} for all image generation. 📝 TRANSPARENT LOGO NOTE: The logo files are PNG format with transparent backgrounds - you can seamlessly integrate them into any background design without worrying about background conflicts. Additionally, you should not change the logo file, you should use the logo file as it is. You can enlarge or shrink the logo but you should never change its layout and structure.",
            "tools": [4, 12]
        }
    }


def _get_evaluation_team_config(team_name: str) -> Dict:
    """get EvaluationTeam config"""
    return {
        "team": "Evaluation",
        "return": team_name,
        "prompt": "Evaluate the generated banner and provide feedback for improvement.",
        "additional_prompt": "Focus on visual appeal, brand consistency, and effectiveness.",
        "BannerEvaluator": {
            "prompt": "You are a Banner Evaluator responsible for assessing the quality and effectiveness of the generated banner. Provide constructive feedback on design, messaging, and overall impact.",
            "tools": [4, 10]
        }
    }


def _get_graphic_revisor_config() -> Dict:
    """get GraphicRevisor config"""
    return {
        "prompt": "You are a Graphic Revisor responsible for making improvements to the banner based on evaluation feedback. Focus on enhancing visual elements while maintaining brand consistency. 🎯 CRITICAL: You MUST use BOTH the most recently generated image AND the original logo when editing. The logo file is in the same directory as the generated images. You must find the logo file (it's the PNG file that doesn't start with 'generated_image_') and include it in your input_filepath parameter. Format: input_filepath=['most_recent_image.png', 'logo.png'].",
        "tools": [4, 12]
    } 