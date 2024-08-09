import gradio as gr
import os
import datetime
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from make_inferences import get_inferences
from make_single_inference import get_single_inference

# Example of loading models
model_dict = {
    "Model 1": "models/model-1.pth",
    "Model 2": "models/model-2.pth",
}

def run_inferences(image_dir, uploaded_file, model, results_folder, make_hair_mask):
    # Handle file upload
    if not model:
        return "Error: Not all fields are filled", None, None
    
    architecture = "DeepLabV3"
    loss = "dice"
    model_name = model_dict[model]

    if uploaded_file and image_dir:
        return "Error: Both an image directory and single image are added. Please just choose one.", None, None
    if uploaded_file:
        # Save the uploaded file to a temporary location
        temp_path = uploaded_file.name
        results_folder = None  # Results folder is not needed for file uploads
        results, mask_name = get_single_inference(model_name, temp_path, architecture, loss, make_hair_mask)
        formatted_results = "Analysis results: \n"
        for item in results.keys():
            formatted_results += f"{item}: {results[item]}\n"
        
        if make_hair_mask:
            return formatted_results, Image.open(temp_path), Image.open(mask_name)
        else:
            return formatted_results, None, None

    else:
        if not results_folder:
            return "Error: Please enter the file name you want to save your results to", None, None
        if not image_dir:
            return "Error: Please enter the path to the folder of leaves you want to inference", None, None
        if not results_folder.endswith(".xlsx"):
            return "Error: Results filename must end with .xlsx", None, None
        if not os.path.exists(image_dir):
            return f"Error: {image_dir} is not a valid path.", None, None
        

    results_folder, mask_dir = get_inferences(model_name, image_dir, architecture, loss, results_folder, make_hair_mask)

    if make_hair_mask:
        return f"Data saved to {results_folder} and generated masks saved to {mask_dir}", None, None
    else:
        return f"Data saved to {results_folder}", None, None

# Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <style>
                .container { 
                    padding: 20px;
                    background-color: #f4f4f4; 
                }
                .title {
                    text-align: center;
                    color: #2a2a2a;
                }
                .gradio-row {
                    margin-bottom: 15px;
                }
            </style>
            # Leaf Hair Analysis: Make Inferences
            """,
            elem_id="header"
        )
        
        with gr.Row(elem_id="inputs-row"):
            with gr.Column():
                image_dir = gr.Textbox(
                    label="Image Directory Path (Optional)", 
                    placeholder="Enter path to the image directory or leave blank",
                    elem_id="image-dir"
                )
                uploaded_file = gr.File(
                    label="Or Upload a File",
                    elem_id="file-upload"
                )
                model = gr.Dropdown(
                    choices=list(model_dict.keys()), 
                    label="Select Model",
                    elem_id="model-select"
                )
                results_folder = gr.Textbox(
                    label="Results File (Required if using image directory)",
                    placeholder="Enter results folder path",
                    elem_id="results-folder"
                )
                make_hair_mask = gr.Checkbox(
                    label="Make Hair Mask",
                    elem_id="hair-mask"
                )
        
        with gr.Row():
            submit_btn = gr.Button("Run Inferences", elem_id="submit-btn")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Result",
                elem_id="result-output"
            )
            output_image = gr.Image(
                label="Result Image (if single image and mask saved)",
                elem_id="result-image"
            )
            output_mask = gr.Image(
                label="Mask Image (if single image and mask saved)",
                elem_id="mask-image"
            )
        
        def handle_submit(image_dir, uploaded_file, model, results_folder, make_hair_mask):
            # Check if only one of image_dir or uploaded_file is provided
            if image_dir and uploaded_file:
                return "Error: Please provide either an image directory or upload a file, not both.", None, None
            if not image_dir and not uploaded_file:
                return "Error: Please provide either an image directory or upload a file.", None, None
            if not model:
                return "Error: Please select a model.", None, None

            return run_inferences(image_dir, uploaded_file, model, results_folder, make_hair_mask)
        
        submit_btn.click(
            handle_submit,
            inputs=[image_dir, uploaded_file, model, results_folder, make_hair_mask],
            outputs=[output_text, output_image, output_mask]
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
