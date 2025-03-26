# app.py - TextDiffuser-2 implementation for Hugging Face Spaces
import os
import torch
import gradio as gr
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleTextDiffuser:
    """
    Simple implementation of TextDiffuser-2 concept for Hugging Face Spaces
    """
    def __init__(self):
        # Load language model for layout generation
        # Using a small model for efficiency
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.language_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.language_model.to(device)
        
        # Only load the diffusion model if we have a GPU
        self.diffusion_model = None
        if torch.cuda.is_available():
            self.diffusion_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            )
            self.diffusion_model.to(device)
        
        print("Models initialized")
    
    def generate_layout(self, prompt, image_size=(512, 512), num_text_elements=3):
        """Generate text layout based on prompt"""
        width, height = image_size
        
        # Format the prompt for layout generation
        layout_prompt = f"""
        Create a layout for an image with:
        - Description: {prompt}
        - Image size: {width}x{height}
        - Number of text elements: {num_text_elements}
        
        Generate text content and positions:
        """
        
        # Generate layout using LM
        input_ids = self.tokenizer.encode(layout_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = self.language_model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 150,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        layout_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse the generated layout (simplified)
        # In a real implementation, this would be more sophisticated
        text_elements = []
        
        # Simple fallback: generate random layout
        import random
        
        # Create a title element
        title = prompt.split()[:5]
        title = " ".join(title) + "..."
        title_x = width // 4
        title_y = height // 4
        text_elements.append({
            "text": title,
            "position": (title_x, title_y),
            "size": 24,
            "color": (0, 0, 0),
            "type": "title"
        })
        
        # Create additional text elements
        sample_texts = [
            "Premium Quality",
            "Best Value",
            "Limited Edition",
            "New Collection",
            "Special Offer",
            "Coming Soon",
            "Best Seller",
            "Top Choice",
            "Featured Product",
            "Exclusive Deal"
        ]
        
        for i in range(1, num_text_elements):
            x = random.randint(width // 8, width * 3 // 4)
            y = random.randint(height // 3, height * 3 // 4)
            text = sample_texts[i % len(sample_texts)]
            color = (
                random.randint(0, 200),
                random.randint(0, 200),
                random.randint(0, 200)
            )
            
            text_elements.append({
                "text": text,
                "position": (x, y),
                "size": 18,
                "color": color,
                "type": f"element_{i}"
            })
        
        return text_elements, layout_text
    
    def generate_image(self, prompt, image_size=(512, 512)):
        """Generate base image using diffusion model or placeholder"""
        width, height = image_size
        
        if self.diffusion_model and torch.cuda.is_available():
            # Generate image using diffusion model
            image = self.diffusion_model(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=30
            ).images[0]
        else:
            # Create a placeholder gradient image
            image = Image.new("RGB", image_size, (240, 240, 240))
            
            # Add a colored gradient background
            for y in range(height):
                for x in range(width):
                    r = int(240 - 100 * (y / height))
                    g = int(240 - 50 * (x / width))
                    b = int(240 - 75 * ((x + y) / (width + height)))
                    image.putpixel((x, y), (r, g, b))
        
        return image
    
    def render_text(self, image, text_elements):
        """Render text elements onto the image"""
        image_with_text = image.copy()
        draw = ImageDraw.Draw(image_with_text)
        
        for element in text_elements:
            try:
                font_size = element["size"]
                
                # Try to load a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("Arial.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()
                
                # Draw text with background for better visibility
                text = element["text"]
                position = element["position"]
                color = element["color"]
                
                # Get text size to create background
                bbox = draw.textbbox(position, text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw semi-transparent background
                padding = 5
                background_box = [
                    position[0] - padding,
                    position[1] - padding,
                    position[0] + text_width + padding,
                    position[1] + text_height + padding
                ]
                draw.rectangle(background_box, fill=(255, 255, 255, 200))
                
                # Draw text
                draw.text(position, text, fill=color, font=font)
                
            except Exception as e:
                print(f"Error rendering text: {e}")
                continue
        
        return image_with_text
    
    def visualize_layout(self, text_elements, image_size=(512, 512)):
        """Create a visualization of the text layout"""
        width, height = image_size
        image = Image.new("RGB", image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw grid
        for x in range(0, width, 50):
            draw.line([(x, 0), (x, height)], fill=(230, 230, 230))
        for y in range(0, height, 50):
            draw.line([(0, y), (width, y)], fill=(230, 230, 230))
        
        # Draw text elements
        for element in text_elements:
            position = element["position"]
            text = element["text"]
            element_type = element.get("type", "unknown")
            
            # Draw position marker
            circle_radius = 5
            circle_bbox = [
                position[0] - circle_radius,
                position[1] - circle_radius,
                position[0] + circle_radius,
                position[1] + circle_radius
            ]
            draw.ellipse(circle_bbox, fill=(255, 0, 0))
            
            # Draw text label
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 12)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw text preview and position info
            info_text = f"{text} ({element_type})"
            pos_text = f"Position: ({position[0]}, {position[1]})"
            draw.text((position[0] + 10, position[1]), info_text, fill=(0, 0, 0), font=font)
            draw.text((position[0] + 10, position[1] + 15), pos_text, fill=(0, 0, 255), font=font)
        
        return image
    
    def generate_text_image(self, prompt, width=512, height=512, num_text_elements=3):
        """Generate an image with rendered text based on prompt"""
        # Validate inputs
        width = max(256, min(1024, width))
        height = max(256, min(1024, height))
        num_text_elements = max(1, min(5, num_text_elements))
        
        image_size = (width, height)
        
        # Step 1: Generate text layout
        text_elements, layout_text = self.generate_layout(prompt, image_size, num_text_elements)
        
        # Step 2: Generate base image
        base_image = self.generate_image(prompt, image_size)
        
        # Step 3: Render text onto the image
        image_with_text = self.render_text(base_image, text_elements)
        
        # Step 4: Create layout visualization
        layout_visualization = self.visualize_layout(text_elements, image_size)
        
        # Step 5: Format layout information for display
        layout_info = {
            "prompt": prompt,
            "image_size": image_size,
            "num_text_elements": num_text_elements,
            "text_elements": text_elements,
            "layout_generation_prompt": layout_text
        }
        
        formatted_layout = json.dumps(layout_info, indent=2)
        
        return image_with_text, layout_visualization, formatted_layout

# Initialize the model
model = SimpleTextDiffuser()

# Define the Gradio interface
def process_request(prompt, width, height, num_text_elements):
    try:
        width = int(width)
        height = int(height)
        num_text_elements = int(num_text_elements)
        
        image, layout, layout_info = model.generate_text_image(
            prompt,
            width=width,
            height=height,
            num_text_elements=num_text_elements
        )
        
        return image, layout, layout_info
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return None, None, error_message

# Create the Gradio app
with gr.Blocks(title="TextDiffuser-2 Demo") as demo:
    gr.Markdown("""
    # TextDiffuser-2 Demo
    
    This demo implements the concepts from the paper "[TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](https://arxiv.org/abs/2311.16465)" by Jingye Chen et al.
    
    Generate images with text by providing a descriptive prompt below.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                value="A modern business poster with company name and tagline",
                lines=3
            )
            
            with gr.Row():
                width_input = gr.Number(label="Width", value=512, minimum=256, maximum=1024, step=64)
                height_input = gr.Number(label="Height", value=512, minimum=256, maximum=1024, step=64)
            
            num_elements_input = gr.Slider(
                label="Number of Text Elements",
                minimum=1,
                maximum=5,
                value=3,
                step=1
            )
            
            submit_button = gr.Button("Generate Image", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Generated Image"):
                    image_output = gr.Image(label="Image with Text")
                
                with gr.TabItem("Layout Visualization"):
                    layout_output = gr.Image(label="Text Layout")
                
                with gr.TabItem("Layout Information"):
                    layout_info_output = gr.Code(language="json", label="Layout Data")
    
    gr.Markdown("""
    ## Example Prompts
    
    Try these prompts or create your own:
    """)
    
    examples = gr.Examples(
        examples=[
            ["A movie poster for a sci-fi thriller", 512, 768, 3],
            ["A motivational quote on a sunset background", 768, 512, 2],
            ["A coffee shop menu with prices", 512, 512, 4],
            ["A modern business card design", 512, 384, 3],
        ],
        inputs=[prompt_input, width_input, height_input, num_elements_input]
    )
    
    submit_button.click(
        fn=process_request,
        inputs=[prompt_input, width_input, height_input, num_elements_input],
        outputs=[image_output, layout_output, layout_info_output]
    )
    
    gr.Markdown("""
    ## About
    
    This is a simplified implementation for demonstration purposes. The full approach described in the paper involves deeper integration of language models with the diffusion process.
    
    Running on: """ + str(device))

# Launch the app
if __name__ == "__main__":
    demo.launch()