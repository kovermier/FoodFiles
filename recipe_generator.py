import os
import base64
import logging
import time
import random
from typing import Optional, Tuple
import json

import gradio as gr
from groq import Groq
from PIL import Image
import re
import tempfile
from functools import partial
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure GROQ_API_KEY is set
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable not set.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Constants
MAX_RETRIES = 5
IMAGE_RESIZE_FACTOR = 1
IMAGE_QUALITY = 85
MAX_TOKENS = {
    'analysis': 750,
    'recipe': 2000,
    'nutrition': 1000,
    'menu': 300,
    'cost': 1000
}

def process_and_encode_image(image: Image.Image) -> str:
    """
    Processes and encodes the image to base64 format suitable for API usage.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            image = image.convert('RGB')
            image = image.resize(
                (
                    int(image.width * IMAGE_RESIZE_FACTOR),
                    int(image.height * IMAGE_RESIZE_FACTOR)
                )
            )
            image.save(temp_path, format="JPEG", optimize=True, quality=IMAGE_QUALITY)
        
        with open(temp_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        os.remove(temp_path)
        logger.info("Image processed and encoded successfully.")
        return base64_image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def groq_api_call(model: str, messages: list, max_tokens: int, json_mode: bool = False) -> str:
    """
    Makes an API call to the Groq model with retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_mode else None
            )
            content = response.choices[0].message.content
            logger.info(f"API call successful using model {model}.")
            return content
        except Exception as e:
            logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = (2 ** attempt) + random.random()
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached. Failing the API call.")
                raise

def analyze_image(base64_image: str) -> str:
    """
    Analyzes the image using the vision model and returns a detailed description.
    """
    prompt = "Our goal is to analyze images of food with as much detail as possible. Analyze the following image of a dish and provide a detailed description including ingredients, cooking techniques, presentation, textures, colors, and any unique elements. The analysis: "

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    analysis = groq_api_call(
        model="llama-3.2-11b-vision-preview",
        messages=messages,
        max_tokens=MAX_TOKENS['analysis']
    )
    return analysis

def generate_recipe(analysis: str) -> str:
    """
    Generates a recipe based on the analysis.
    """
    prompt = f"""Based on the following analysis, generate a detailed recipe including:

1. Recipe Name
2. List of ingredients with approximate quantities
3. Step-by-step instructions
4. Variations or substitutions
5. Serving suggestions
6. Menu description suitable for a restaurant

**Format the recipe exactly as follows, using the same section headings and Markdown syntax:**

# **[Recipe Name]**

## **Ingredients**
- [Ingredient 1]
- [Ingredient 2]
...

## **Instructions**
1. [Step 1]
2. [Step 2]
...

## **Variations**
- [Variation 1]
- [Variation 2]
...

## **Serving Suggestions**
- [Suggestion 1]
- [Suggestion 2]
...

## **Menu Description**
[Menu description here]

Analysis:
{analysis}
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    recipe = groq_api_call(
        model="mixtral-8x7b-32768",
        messages=messages,
        max_tokens=MAX_TOKENS['recipe']
    )
    return recipe

def parse_recipe(recipe_markdown: str) -> dict:
    """
    Parses the recipe Markdown into its components.
    """
    recipe_parts = {}
    
    # Extract recipe name
    recipe_name_match = re.search(r"# \*\*(.+?)\*\*", recipe_markdown)
    recipe_parts['recipe_name'] = recipe_name_match.group(1).strip() if recipe_name_match else "Unknown Recipe"

    # Sections to extract
    sections = ['Ingredients', 'Instructions', 'Variations', 'Serving Suggestions', 'Menu Description']
    
    for i, section in enumerate(sections):
        pattern = rf"## \*\*{section}\*\*\n(.*?)(?=## \*\*{'|'.join(sections[i+1:])}\*\*|$)"
        match = re.search(pattern, recipe_markdown, re.DOTALL)
        recipe_parts[section.lower().replace(' ', '_')] = match.group(1).strip() if match else ""
    return recipe_parts

def get_nutritional_info(ingredients: str) -> str:
    """
    Provides nutritional analysis for the given ingredients in a nutrition label format.
    """
    if not ingredients.strip():
        return "Ingredients not provided."

    prompt = f"""You are a nutrition expert API that provides nutritional analysis for given ingredients. Respond with a JSON object containing the nutritional information for the entire dish.

Analyze the following ingredients and provide nutritional information for the complete dish:

{ingredients}

The JSON response should have the following structure:
{{
    "serving_size": "X g",
    "calories": int,
    "total_fat": "X g",
    "saturated_fat": "X g",
    "trans_fat": "X g",
    "cholesterol": "X mg",
    "sodium": "X mg",
    "total_carbohydrate": "X g",
    "dietary_fiber": "X g",
    "total_sugars": "X g",
    "protein": "X g",
    "vitamin_d": "X mcg",
    "calcium": "X mg",
    "iron": "X mg",
    "potassium": "X mg"
}}

Provide a single set of nutritional information for the entire dish, not individual ingredients. If you cannot provide exact values, use reasonable estimates based on similar dishes. Ensure all fields are present in the response, using "0" or "0 g" for any nutrients that are not applicable or present in negligible amounts.
"""

    messages = [
        {"role": "system", "content": "You are a nutrition expert API that responds in JSON format."},
        {"role": "user", "content": prompt}
    ]

    try:
        nutrition_response = groq_api_call(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=MAX_TOKENS['nutrition'],
            json_mode=True
        )
        
        nutrition_data = json.loads(nutrition_response)
        logger.info("Successfully parsed nutrition data JSON.")

        # Ensure we have a single set of nutritional information
        if isinstance(nutrition_data, dict) and 'serving_size' in nutrition_data:
            nutrition_data = {'dish': nutrition_data}
        elif not any('serving_size' in item for item in nutrition_data.values()):
            raise ValueError("Invalid nutrition data format")

        # Combine multiple ingredients if necessary
        combined_nutrition = combine_nutrition_data(nutrition_data)

        nutrition_label = generate_nutrition_label(combined_nutrition)
        return nutrition_label
    except Exception as e:
        logger.error(f"Error in get_nutritional_info: {e}")
        return f"Error parsing nutritional information: {e}"

def combine_nutrition_data(nutrition_data: dict) -> dict:
    """
    Combines nutritional information from multiple ingredients into a single set of data.
    """
    combined = {
        "serving_size": "1 serving",
        "calories": 0,
        "total_fat": 0,
        "saturated_fat": 0,
        "trans_fat": 0,
        "cholesterol": 0,
        "sodium": 0,
        "total_carbohydrate": 0,
        "dietary_fiber": 0,
        "total_sugars": 0,
        "protein": 0,
        "vitamin_d": 0,
        "calcium": 0,
        "iron": 0,
        "potassium": 0
    }

    for item in nutrition_data.values():
        for key, value in item.items():
            if key == "serving_size":
                continue
            if isinstance(value, str):
                # Remove 'g' or other units before conversion
                value = float(re.sub(r'[^\d.]+', '', value))
            combined[key] += value

    # Convert back to strings with units
    for key, value in combined.items():
        if key == "calories":
            combined[key] = int(value)
        elif key != "serving_size":
            unit = next((item[key].split()[-1] for item in nutrition_data.values() if key in item), "g")
            combined[key] = f"{value:.1f} {unit}"

    return combined

def generate_nutrition_label(nutrition_data: dict) -> str:
    """
    Generates an HTML nutrition label from the given nutritional data.
    """
    nutrition_label = f"""
    <div class="nutrition-label">
        <p class="nutrition-footnote">Please Note that these are estimations.</p
        <h2 class="nutrition-title">Nutrition Facts</h2>
        <p class="nutrition-serving">Serving Size {nutrition_data['serving_size']}</p>
        <div class="nutrition-calories">
            <span class="left">Calories</span>
            <span class="right">{nutrition_data['calories']}</span>
        </div>
        <div class="nutrition-percent">% Daily Value*</div>
        <div class="nutrition-item">
            <span class="left"><strong>Total Fat</strong> {nutrition_data['total_fat']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['total_fat'], 78)}%</span>
        </div>
        <div class="nutrition-sub-item">
            <span class="left">Saturated Fat {nutrition_data['saturated_fat']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['saturated_fat'], 20)}%</span>
        </div>
        <div class="nutrition-sub-item">
            <span class="left"><em>Trans</em> Fat {nutrition_data['trans_fat']}</span>
        </div>
        <div class="nutrition-item">
            <span class="left"><strong>Cholesterol</strong> {nutrition_data['cholesterol']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['cholesterol'], 300)}%</span>
        </div>
        <div class="nutrition-item">
            <span class="left"><strong>Sodium</strong> {nutrition_data['sodium']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['sodium'], 2300)}%</span>
        </div>
        <div class="nutrition-item">
            <span class="left"><strong>Total Carbohydrate</strong> {nutrition_data['total_carbohydrate']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['total_carbohydrate'], 275)}%</span>
        </div>
        <div class="nutrition-sub-item">
            <span class="left">Dietary Fiber {nutrition_data['dietary_fiber']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['dietary_fiber'], 28)}%</span>
        </div>
        <div class="nutrition-sub-item">
            <span class="left">Total Sugars {nutrition_data['total_sugars']}</span>
        </div>
        <div class="nutrition-sub-item">
            <span class="left">Includes 0g Added Sugars</span>
            <span class="right">0%</span>
        </div>
        <div class="nutrition-item">
            <span class="left"><strong>Protein</strong> {nutrition_data['protein']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['protein'], 50)}%</span>
        </div>
        <div class="nutrition-item">
            <span class="left">Vitamin D {nutrition_data['vitamin_d']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['vitamin_d'], 20)}%</span>
        </div>
        <div class="nutrition-item">
            <span class="left">Calcium {nutrition_data['calcium']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['calcium'], 1300)}%</span>
        </div>
        <div class="nutrition-item">
            <span class="left">Iron {nutrition_data['iron']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['iron'], 18)}%</span>
        </div>
        <div class="nutrition-item">
            <span class="left">Potassium {nutrition_data['potassium']}</span>
            <span class="right">{calculate_daily_value(nutrition_data['potassium'], 4700)}%</span>
        </div>
        <p class="nutrition-footnote">* The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a daily diet. 2,000 calories a day is used for general nutrition advice.</p>
    </div>
    """
    return nutrition_label

def calculate_daily_value(value_str: str, daily_value: int) -> int:
    """
    Calculate the percentage of daily value for a nutrient.
    """
    try:
        # Extract numeric value using regex
        numeric_value = re.search(r'\d+\.?\d*', value_str)
        if numeric_value:
            value = float(numeric_value.group())
            return round((value / daily_value) * 100)
        else:
            return 0
    except Exception as e:
        logger.error(f"Error calculating daily value: {e}")
        return 0

def estimate_cost(ingredients: str) -> str:
    """
    Estimates the cost of the ingredients.
    """
    if not ingredients.strip():
        return "Ingredients not provided."

    prompt = f"""You are a personal shopper and restaurant logistician. Estimate the total cost to produce the dish using the following ingredients. Provide a succint summary breakdown of costs for the recipe in question.

Ingredients:
{ingredients}

Assume average market prices.
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    cost = groq_api_call(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=MAX_TOKENS['cost']
    )
    return cost

def create_menu_description(recipe: str) -> str:
    """
    Creates an enticing menu description based on the recipe.
    """
    prompt = f"""You are a restauranteur and influencer; Based on the following recipe, create an enticing menu description suitable for a restaurant.

Recipe:
{recipe}

The description should be 2-3 sentences.
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    menu_description = groq_api_call(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=MAX_TOKENS['menu']
    )
    return menu_description

def process_image_and_generate_recipe(image: Optional[Image.Image]) -> Tuple[str, str, str, str, str, str, str]:
    """
    Main function that processes the image and generates all outputs.
    """
    if image is None:
        return ("No image uploaded.", "", "", "", "", "", "")
    try:
        logger.info("Starting the recipe generation process.")
        base64_image = process_and_encode_image(image)
        analysis = analyze_image(base64_image)
        recipe_markdown = generate_recipe(analysis)
        logger.info(f"Generated Recipe:\n{recipe_markdown}")
        recipe_parts = parse_recipe(recipe_markdown)
        ingredients = recipe_parts.get('ingredients', '')
        logger.info(f"Extracted Ingredients:\n{ingredients}")
        
        try:
            nutrition = get_nutritional_info(ingredients)
        except Exception as e:
            logger.error(f"Error getting nutritional info: {e}")
            nutrition = "Error retrieving nutritional information."
        
        try:
            cost = estimate_cost(ingredients)
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            cost = "Error estimating cost."
        
        menu_description = recipe_parts.get('menu_description', '')
        instructions = recipe_parts.get('instructions', '')

        return (
            analysis,
            recipe_markdown,
            nutrition,
            cost,
            menu_description,
            ingredients,
            instructions
        )
    except Exception as e:
        error_message = f"An error occurred: {e}"
        logger.error(error_message)
        return (error_message, "", "", "", "", "", "")

# Custom CSS for nutrition label only
custom_css = """
.nutrition-label {
    border: 2px solid #333;
    padding: 15px;
    width: 100%;
    max-width: 400px;
    font-family: 'Arial', sans-serif;
    background-color: white;
    box-sizing: border-box;
    color: black !important;
    margin: 0 auto;
}
.nutrition-label * {
    color: black !important;
}
.nutrition-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
    border-bottom: 10px solid #333;
    padding-bottom: 10px;
}
.nutrition-serving {
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 15px;
    border-bottom: 1px solid #333;
    padding-bottom: 10px;
}
.nutrition-calories {
    font-size: 32px;
    font-weight: bold;
    border-bottom: 10px solid #333;
    padding: 10px 0;
    margin-bottom: 10px;
}
.nutrition-percent {
    font-size: 14px;
    font-weight: bold;
    text-align: right;
    border-bottom: 1px solid #333;
    padding-bottom: 5px;
    margin-bottom: 10px;
}
.nutrition-item, .nutrition-sub-item {
    font-size: 14px;
    border-bottom: 1px solid #333;
    padding: 5px 0;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
}
.nutrition-item {
    font-weight: bold;
}
.nutrition-sub-item {
    padding-left: 20px;
}
.left {
    flex: 1;
    margin-right: 10px;
}
.right {
    text-align: right;
}
.nutrition-footnote {
    font-size: 10px;
    margin-top: 15px;
    line-height: 1.4;
}
"""

# Add this before the Gradio interface definition
def validate_api_key(api_key: str) -> bool:
    """Validates Groq API key format"""
    if not api_key:
        return False
    return api_key.startswith("gsk_") and len(api_key) >= 40

# Modify the process function to handle state
def process_with_key(image: Optional[Image.Image], 
                    api_key: str,
                    progress: gr.Progress = gr.Progress()) -> Tuple[str, str, str, str, str, str, str, str]:
    """Processes image with API key and shows progress"""
    if not validate_api_key(api_key):
        return ("Please enter a valid Groq API Key (starts with 'gsk_').", "", "", "", "", "", "", "❌ Invalid API key")
    
    try:
        client = Groq(api_key=api_key)
        
        if image is None:
            return ("No image uploaded.", "", "", "", "", "", "", "⚠️ No image provided")
            
        progress(0, desc="Processing image...")
        base64_image = process_and_encode_image(image)
        
        progress(0.2, desc="Analyzing image...")
        analysis = analyze_image(base64_image)
        
        progress(0.4, desc="Generating recipe...")
        recipe_markdown = generate_recipe(analysis)
        recipe_parts = parse_recipe(recipe_markdown)
        ingredients = recipe_parts.get('ingredients', '')
        
        progress(0.6, desc="Calculating nutrition...")
        try:
            nutrition = get_nutritional_info(ingredients)
        except Exception as e:
            logger.error(f"Error getting nutritional info: {e}")
            nutrition = "Error retrieving nutritional information."
        
        progress(0.8, desc="Estimating costs...")
        try:
            cost = estimate_cost(ingredients)
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            cost = "Error estimating cost."
        
        menu_description = recipe_parts.get('menu_description', '')
        instructions = recipe_parts.get('instructions', '')

        progress(1.0, desc="Done!")
        return (
            analysis,
            recipe_markdown,
            nutrition,
            cost,
            menu_description or "No menu description available.",
            ingredients,
            instructions,
            "✅ Recipe generated successfully!"
        )
    except Exception as e:
        error_message = f"An error occurred: {e}"
        logger.error(error_message)
        return (error_message, "", "", "", "", "", "", "❌ Error occurred")

# Modified Gradio interface
with gr.Blocks(theme=gr.themes.Ocean(), css=custom_css) as demo:
    # Store API key state
    api_key_state = gr.State("")
    
    with gr.Row():
        gr.HTML("""
<a href="https://www.smartbrandstrategies.com?utm_source=foodfiles&utm_medium=app&utm_campaign=recipe_generator" target="_blank" style="text-decoration: none;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 100" style="width: 350px; height: 50px;">
    <circle cx="50" cy="50" r="45" fill="#2DD4BF" opacity="0.2"/>
    <circle cx="50" cy="50" r="35" fill="#2DD4BF" opacity="0.4"/>
    <path d="M30 30 L30 70 M50 30 L50 70 M70 30 L70 70" 
            stroke="#2DD4BF" 
            stroke-width="8" 
            stroke-linecap="round"/>
    <text x="100" y="65" 
            font-family="Arial" 
            font-size="48" 
            font-weight="bold" 
            fill="#2DD4BF">
        FoodFiles
    </text>
    <text x="330" y="65"
            font-family="Arial"
            font-size="20"
            fill="#2DD4BF">
        by SmartBrandStrategies.com
    </text>
</svg>
</a>
""")
    
    # API Key section with Groq badge in same row
    with gr.Row():
        with gr.Column(scale=3):
            api_key_input = gr.Textbox(
                label="Enter your Groq API Key", 
                type="password",
                placeholder="gsk_...",
                value=""
            )
        with gr.Column(scale=1):
            gr.HTML("""<a href="https://groq.com" target="_blank" rel="noopener noreferrer">
                <img src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg" 
                    alt="Powered by Groq for fast inference."
                    style="margin: 10px 0; max-width: 200px;"/>
            </a>""")
    
    with gr.Row():
        gr.Markdown("""To get your API key:
1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key (don't share with anyone)
5. Upload an image of a dish to get an analysis and likely recipe and more!""")
    
    # Add loading indicator
    with gr.Row():
        status_message = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Dish Image")
            submit_button = gr.Button("Generate Recipe", variant="primary")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Analysis"):
                    analysis_output = gr.Markdown(label="Image Analysis")
                with gr.TabItem("Full Recipe"):
                    recipe_output = gr.Markdown(label="Full Recipe")
                with gr.TabItem("Nutritional Information"):
                    nutrition_output = gr.HTML(label="Nutritional Information")
                with gr.TabItem("Cost Estimation"):
                    cost_output = gr.Markdown(label="Cost Estimation")
                with gr.TabItem("Menu Description"):
                    menu_output = gr.Markdown(label="Menu Description")
                with gr.TabItem("Ingredients"):
                    ingredients_output = gr.Markdown(label="Ingredients")
                with gr.TabItem("Instructions"):
                    instructions_output = gr.Markdown(label="Instructions")

    # Function to save API key to state
    def save_api_key(api_key):
        if validate_api_key(api_key):
            return "✅ API key saved", api_key
        return "❌ Invalid API key format", ""

    # Wire up the components
    api_key_input.change(
        fn=save_api_key,
        inputs=[api_key_input],
        outputs=[status_message, api_key_state]
    )

    submit_button.click(
        fn=process_with_key,
        inputs=[
            image_input,
            api_key_input
        ],
        outputs=[
            analysis_output,
            recipe_output,
            nutrition_output,
            cost_output,
            menu_output,
            ingredients_output,
            instructions_output,
            status_message
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True, debug=True)
