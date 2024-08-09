import gradio as gr
import os
import datetime
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from make_inferences import get_inferences

# Load your pre-trained model
# def load_model(model_path):
#     model = torch.load(model_path)
#     return model

# Example of loading models
model_dict = {
    "Model 1": "models/model-1.pth",
    "Model 2": "models/model-2.pth",
}

def run_inference(image_dir, model, results_folder, make_hair_mask):

    if not image_dir or not results_folder or not model:
        return "Error: Not all fields are filled"
    if not results_folder.endswith(".xlsx"):
        return "Error: Results filename must end with .xlsx"
    
    if not os.path.exists(image_dir):
        return f"Error: {image_dir} is not a valid path. Try repository06032024_DM_6-8-2024_3dpi_1"
    
    architecture = "DeepLabV3"
    loss = "dice"
    model_name = model_dict[model]

    results_folder, mask_dir = get_inferences(model_name, image_dir, architecture, loss, results_folder, make_hair_mask)

    if make_hair_mask:
        return f"Data saved to {results_folder} and generated masks saved to {mask_dir}"
    else:
        return f"Data saved to {results_folder}"

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
                    label="Image Directory Path", 
                    placeholder="Enter path to the image directory",
                    elem_id="image-dir"
                )
                model = gr.Dropdown(
                    choices=list(model_dict.keys()), 
                    label="Select Model",
                    elem_id="model-select"
                )
                results_folder = gr.Textbox(
                    label="Results File", 
                    placeholder="Enter results folder path",
                    elem_id="results-folder"
                )
                make_hair_mask = gr.Checkbox(
                    label="Make Hair Mask",
                    elem_id="hair-mask"
                )
        
        with gr.Row():
            submit_btn = gr.Button("Run Inferences", elem_id="submit-btn")
        
        output = gr.Textbox(
            label="Result",
            elem_id="result-output"
        )
        
        submit_btn.click(
            run_inference,
            inputs=[image_dir, model, results_folder, make_hair_mask],
            outputs=output
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
