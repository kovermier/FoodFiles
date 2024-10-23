# FoodFiles - AI-Powered Recipe Generator

FoodFiles is a comprehensive web application demonstrating modern AI capabilities in culinary applications. Using computer vision and multiple AI models, it analyzes food images to generate detailed potential recipes, nutritional information, cost estimates, and professional menu descriptions. 

The application is intentionally structured as a single module to showcase CSS integration patterns and function interactions, making it an excellent learning resource for developers. Powered by Groq's fast inference capabilities and developed by Kurt Overmier at SmartBrandStrategies.com, this project serves as both a functional tool and educational reference.

## Features

- Image Analysis: Detailed analysis of food images identifying ingredients, cooking techniques, and presentation
- Recipe Generation: Complete recipes with ingredients, instructions, and variations
- Nutritional Information: Detailed nutrition facts label with daily value percentages
- Cost Estimation: Ingredient cost breakdown and total recipe cost
- Menu Description: Restaurant-style dish descriptions
- Progress Tracking: Real-time progress indicators during generation
- Mobile-Friendly Interface: Responsive design that works on all devices
- Secure API Key Handling: Input your Groq API key directly in the interface

## Prerequisites

- Python 3.7+
- Required Python packages:
  - groq
  - gradio
  - Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/foodfiles.git
cd foodfiles
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python recipe_generator.py
```

2. Open your browser and navigate to `http://127.0.0.1:7860`

3. Get your Groq API key:
   - Visit Groq Console
   - Sign up or log in
   - Go to API Keys section
   - Create a new API key

4. Enter your Groq API key in the application interface
5. Upload an image of a dish
6. Click "Generate Recipe" to analyze the image and generate results

## API Models Used

- `llama-3.2-11b-vision-preview`: Image analysis
- `mixtral-8x7b-32768`: Recipe generation
- `llama3-8b-8192`: Nutritional analysis
- `llama-3.1-8b-instant`: Cost estimation and menu descriptions

## Configuration

Key configuration constants in `recipe_generator.py`:
```python
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
```

## Error Handling

- Automatic retry logic for API calls
- Exponential backoff between retries
- Comprehensive error logging
- User-friendly error messages in the interface
- API key validation and status feedback

## Development

The application uses:

- Gradio for the web interface
- PIL for image processing
- Custom CSS for nutrition label styling
- Markdown formatting for recipe output
- Regex for recipe parsing
- JSON mode for structured data handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [License](LICENSE) file for details.

Copyright (c) 2024 SmartBrandStrategies.com - Kurt Overmier

## Acknowledgments

- Powered by Groq for fast inference
- Built with Gradio
