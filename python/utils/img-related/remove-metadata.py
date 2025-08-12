# script to remove metadata from images and add a black bar with text at the bottom
# Peter van Lunteren, 24 Oct, 2024

# conda activate ecoassistcondaenv-base && python /Users/peter/Documents/scripting/sorted-scripts/python/utils/img-related/remove-metadata.py


import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from fontTools.ttLib import TTFont

# Specify the input and output folders
input_folder = "/Users/peter/Desktop/test-imgs"
output_folder = "/Users/peter/Desktop/test-imgs-clean"

def load_font(font_path, size):
    # Load the font using fontTools
    font = TTFont(font_path)
    # Extract the font into a Pillow-compatible format
    return ImageFont.truetype(font_path, size)

def remove_metadata_and_blacken_bar(input_folder, output_folder, bar_height=65, text="(example image - metadata removed)"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder recursively
    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)

            # Create any necessary subdirectories in the output folder
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Open the image and remove metadata
                with Image.open(input_path) as img:
                    # Get image dimensions
                    width, height = img.size

                    # Create a black rectangle at the bottom
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([0, height - bar_height, width, height], fill="black")

                    # Add the text in the bottom middle of the image
                    # Create an overlay for the text
                    txt_overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                    draw_text = ImageDraw.Draw(txt_overlay)

                    # Load a custom font and set a larger font size
                    font_path = "/Library/Fonts/Arial Unicode.ttf"  # Update with your font path
                    font_size = 50  # Adjust the font size as needed
                    font = load_font(font_path, font_size)

                    # Get the size of the text
                    text_bbox = draw_text.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # Calculate the position for the text to be in the bottom center
                    text_x = (width - text_width) / 2
                    text_y = height - bar_height + (bar_height - text_height) / 2  - 15 # Centered vertically in the bar

                    # Add the text to the overlay in white color
                    draw_text.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))  # White color

                    # Merge the overlay with the original image
                    img = Image.alpha_composite(img.convert("RGBA"), txt_overlay).convert("RGB")

                    # Save the image without metadata
                    img.save(output_path, format="JPEG")
                print(f"Processed {input_path} -> {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")


# Run the function
remove_metadata_and_blacken_bar(input_folder, output_folder)
