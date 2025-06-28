#!/usr/bin/env python3
"""
UVR Web Interface using Gradio
Provides a web-based interface for Ultimate Vocal Remover
"""

import gradio as gr
import os
import sys
import tempfile
import traceback
from pathlib import Path
import shutil

# Add the current directory to the path so we can import from the UVR modules
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_PATH)
os.chdir(BASE_PATH)

# Import UVR modules
from UVR import ModelData
from separate import SeperateVR, SeperateMDX, SeperateDemucs, save_format
from gui_data.constants import *

class UVRWeb:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.available_models = self.scan_models()
        
    def scan_models(self):
        """Scan for available models"""
        models = {}
        
        # VR Models
        vr_models_dir = os.path.join(BASE_PATH, "models", "VR_Models")
        if os.path.exists(vr_models_dir):
            vr_models = [f for f in os.listdir(vr_models_dir) if f.endswith('.pth')]
            if vr_models:
                models['VR'] = vr_models
                
        # MDX Models  
        mdx_models_dir = os.path.join(BASE_PATH, "models", "MDX_Models")
        if os.path.exists(mdx_models_dir):
            mdx_models = [f for f in os.listdir(mdx_models_dir) if f.endswith('.onnx')]
            if mdx_models:
                models['MDX'] = mdx_models
                
        # Demucs Models
        demucs_models_dir = os.path.join(BASE_PATH, "models", "Demucs_Models")
        if os.path.exists(demucs_models_dir):
            demucs_models = [f for f in os.listdir(demucs_models_dir) if f.endswith('.yaml')]
            if demucs_models:
                models['Demucs'] = demucs_models
                
        return models
    
    def get_model_list(self, model_type):
        """Get list of models for a specific type"""
        return self.available_models.get(model_type, [])
    
    def separate_audio(self, audio_file, model_type, model_name, output_format="wav"):
        """Main separation function"""
        if not audio_file:
            return None, None, "Please upload an audio file"
            
        try:
            # Create temporary output directory
            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the input file path
            input_path = audio_file.name if hasattr(audio_file, 'name') else audio_file
            audio_base = os.path.splitext(os.path.basename(input_path))[0]
            
            # Create ModelData instance
            model_path = os.path.join(BASE_PATH, "models", f"{model_type}_Models", model_name)
            
            # Create a mock process_data dictionary
            process_data = {
                'audio_file': input_path,
                'audio_file_base': audio_base,
                'export_path': output_dir,
                'cached_source_callback': None,
                'cached_model_source_holder': {},
                'is_4_stem_ensemble': False,
                'list_all_models': [],
                'process_iteration': 0,
                'set_progress_bar': lambda x, y: None,  # Mock progress bar
                'write_to_console': lambda x: print(x)   # Mock console
            }
            
            # Initialize model data
            model_data = ModelData(model_path, model_type)
            
            # Select appropriate separator based on model type
            if model_type == "VR":
                separator = SeperateVR(model_data, process_data)
            elif model_type == "MDX":
                separator = SeperateMDX(model_data, process_data)
            elif model_type == "Demucs":
                separator = SeperateDemucs(model_data, process_data)
            else:
                return None, None, f"Unsupported model type: {model_type}"
            
            # Perform separation
            separator.seperate()
            
            # Find output files
            vocals_file = os.path.join(output_dir, f"{audio_base}_(Vocals).{output_format}")
            instrumental_file = os.path.join(output_dir, f"{audio_base}_(Instrumental).{output_format}")
            
            # Check if files exist
            if os.path.exists(vocals_file) and os.path.exists(instrumental_file):
                return vocals_file, instrumental_file, "Separation completed successfully!"
            else:
                # Look for any output files
                output_files = os.listdir(output_dir)
                if output_files:
                    return None, None, f"Separation completed but output format may differ. Files: {output_files}"
                else:
                    return None, None, "Separation failed - no output files generated"
                    
        except Exception as e:
            error_msg = f"Error during separation: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, None, error_msg
    
    def update_model_dropdown(self, model_type):
        """Update model dropdown based on selected type"""
        models = self.get_model_list(model_type)
        if models:
            return gr.update(choices=models, value=models[0])
        else:
            return gr.update(choices=[], value=None)

def create_interface():
    """Create the Gradio interface"""
    uvr = UVRWeb()
    
    with gr.Blocks(title="Ultimate Vocal Remover - Web Interface") as iface:
        gr.Markdown("# Ultimate Vocal Remover - Web Interface")
        gr.Markdown("Upload an audio file and select a model to separate vocals from instrumentals.")
        
        with gr.Row():
            with gr.Column():
                # Input section
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    format="wav"
                )
                
                model_type = gr.Radio(
                    choices=list(uvr.available_models.keys()) if uvr.available_models else ["No models found"],
                    label="Model Type",
                    value=list(uvr.available_models.keys())[0] if uvr.available_models else None
                )
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=uvr.get_model_list(list(uvr.available_models.keys())[0]) if uvr.available_models else [],
                    value=None
                )
                
                output_format = gr.Radio(
                    choices=["wav", "mp3", "flac"],
                    label="Output Format",
                    value="wav"
                )
                
                separate_btn = gr.Button("Separate Audio", variant="primary")
                
            with gr.Column():
                # Output section
                status_text = gr.Textbox(label="Status", interactive=False)
                vocals_output = gr.Audio(label="Vocals", type="filepath")
                instrumental_output = gr.Audio(label="Instrumental", type="filepath")
        
        # Event handlers
        model_type.change(
            fn=uvr.update_model_dropdown,
            inputs=[model_type],
            outputs=[model_name]
        )
        
        separate_btn.click(
            fn=uvr.separate_audio,
            inputs=[audio_input, model_type, model_name, output_format],
            outputs=[vocals_output, instrumental_output, status_text]
        )
    
    return iface

if __name__ == "__main__":
    # Check if models exist
    if not os.path.exists(os.path.join(BASE_PATH, "models")):
        print("Warning: No models directory found. Please ensure you have models installed.")
        print("Expected model directories:")
        print("- models/VR_Models/ (for .pth files)")
        print("- models/MDX_Models/ (for .onnx files)")  
        print("- models/Demucs_Models/ (for .yaml files)")
    
    # Create and launch interface
    iface = create_interface()
    iface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        inbrowser=False         # Don't auto-open browser on server
    )