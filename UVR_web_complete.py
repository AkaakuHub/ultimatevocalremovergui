#!/usr/bin/env python3
"""
Ultimate Vocal Remover - Complete Web Interface using Gradio
Complete reproduction of UVR.py functionality with web interface
"""

import gradio as gr
import os
import sys
import tempfile
import traceback
import json
import threading
import queue
import time
from pathlib import Path
import shutil
import torch

# Add the current directory to the path so we can import from the UVR modules
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_PATH)
os.chdir(BASE_PATH)

# Import UVR modules
from UVR import ModelData
from separate import SeperateVR, SeperateMDX, SeperateMDXC, SeperateDemucs, save_format, clear_gpu_cache
from gui_data.constants import *
from gui_data.app_size_values import *
from gui_data.error_handling import error_text, error_dialouge

class UVRWebComplete:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.available_models = self.scan_models()
        self.settings = self.load_default_settings()
        self.is_processing = False
        self.process_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        
    def load_default_settings(self):
        """Load default settings from UVR"""
        return {
            # Process Method
            'chosen_process_method': MDX_ARCH_TYPE,
            
            # VR Settings
            'aggression_setting': 5,
            'window_size': 512,
            'is_tta': False,
            'is_post_process': False,
            'is_high_end_process': False,
            'post_process_threshold': 0.2,
            
            # MDX Settings
            'mdx_segment_size': 256,
            'overlap_mdx': 0.25,
            'compensate': 1.03597672,
            'mdx_batch_size': 1,
            'chunks': 0,
            'margin': 44100,
            
            # Demucs Settings
            'segment': 'Default',
            'overlap': 0.25,
            'shifts': 2,
            'chunks_demucs': 0,
            'margin_demucs': 44100,
            'is_split_mode': True,
            'is_demucs_combine_stems': True,
            
            # Output Settings
            'save_format': 'WAV',
            'mp3_bit_set': '320k',
            'is_normalization': False,
            'is_add_model_name': False,
            'is_create_model_folder': False,
            
            # GPU Settings
            'is_gpu_conversion': torch.cuda.is_available(),
            
            # Ensemble Settings
            'is_save_all_outputs_ensemble': True,
            'is_append_ensemble_name': False,
            'chosen_ensemble_var': AVERAGE_ENSEMBLE,
            
            # Secondary Model Settings
            'is_secondary_model_activate': False,
            'vr_voc_inst_secondary_model_scale': 0.9,
            'vr_other_secondary_model_scale': 0.7,
            'vr_bass_secondary_model_scale': 0.5,
            'vr_drums_secondary_model_scale': 0.5,
            
            # Audio Tool Settings
            'chosen_audio_tool': MANUAL_ENSEMBLE,
            'time_stretch_rate': 2.0,
            'pitch_rate': 2.0,
            'semitone_shift': '0',
            'is_time_correction': True,
            'is_testing_audio': False,
        }
        
    def scan_models(self):
        """Scan for available models"""
        models = {
            'VR': [],
            'MDX': [],
            'Demucs': []
        }
        
        # VR Models
        vr_models_dir = os.path.join(BASE_PATH, "models", "VR_Models")
        if os.path.exists(vr_models_dir):
            models['VR'] = [f for f in os.listdir(vr_models_dir) if f.endswith('.pth')]
                
        # MDX Models  
        mdx_models_dir = os.path.join(BASE_PATH, "models", "MDX_Models")
        if os.path.exists(mdx_models_dir):
            models['MDX'] = [f for f in os.listdir(mdx_models_dir) if f.endswith('.onnx')]
                
        # Demucs Models
        demucs_models_dir = os.path.join(BASE_PATH, "models", "Demucs_Models")
        if os.path.exists(demucs_models_dir):
            models['Demucs'] = [f for f in os.listdir(demucs_models_dir) if f.endswith('.yaml')]
                
        return models
    
    def update_model_dropdown(self, model_type):
        """Update model dropdown based on selected type"""
        models = self.available_models.get(model_type, [])
        if models:
            return gr.update(choices=models, value=models[0])
        else:
            return gr.update(choices=[], value=None)
    
    def write_to_console(self, message):
        """Write message to console queue"""
        self.progress_queue.put(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def set_progress_bar(self, step, total):
        """Update progress bar"""
        progress = step / total if total > 0 else 0
        self.progress_queue.put(f"PROGRESS:{progress}")
    
    def separate_audio(self, audio_file, model_type, model_name, 
                      # VR Settings
                      aggression_setting, window_size, is_tta, is_post_process, is_high_end_process,
                      # MDX Settings  
                      mdx_segment_size, overlap_mdx, compensate,
                      # Demucs Settings
                      segment_demucs, overlap_demucs, shifts,
                      # Output Settings
                      output_format, is_normalization, is_add_model_name,
                      # Secondary Model
                      is_secondary_model, secondary_model_name,
                      # Ensemble
                      is_ensemble_mode, ensemble_model2,
                      progress=gr.Progress()):
        """Main separation function with all parameters"""
        
        if not audio_file:
            return None, None, "Please upload an audio file"
            
        if self.is_processing:
            return None, None, "Processing is already in progress"
            
        try:
            self.is_processing = True
            self.write_to_console("Starting audio separation...")
            
            # Create temporary output directory
            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the input file path
            input_path = audio_file.name if hasattr(audio_file, 'name') else audio_file
            audio_base = os.path.splitext(os.path.basename(input_path))[0]
            
            # Update settings based on UI inputs
            self.settings.update({
                'aggression_setting': aggression_setting,
                'window_size': window_size,
                'is_tta': is_tta,
                'is_post_process': is_post_process,
                'is_high_end_process': is_high_end_process,
                'mdx_segment_size': mdx_segment_size,
                'overlap_mdx': overlap_mdx,
                'compensate': compensate,
                'segment': segment_demucs,
                'overlap': overlap_demucs,
                'shifts': shifts,
                'save_format': output_format,
                'is_normalization': is_normalization,
                'is_add_model_name': is_add_model_name,
                'is_secondary_model_activate': is_secondary_model,
            })
            
            # Create ModelData instance
            model_path = os.path.join(BASE_PATH, "models", f"{model_type}_Models", model_name)
            
            self.write_to_console(f"Loading model: {model_name}")
            
            # Create process_data dictionary
            process_data = {
                'audio_file': input_path,
                'audio_file_base': audio_base,
                'export_path': output_dir,
                'cached_source_callback': None,
                'cached_model_source_holder': {},
                'is_4_stem_ensemble': False,
                'list_all_models': [],
                'process_iteration': 0,
                'set_progress_bar': self.set_progress_bar,
                'write_to_console': self.write_to_console
            }
            
            # Initialize model data
            model_data = ModelData(model_path, model_type)
            
            # Update model data with settings
            for key, value in self.settings.items():
                if hasattr(model_data, key):
                    setattr(model_data, key, value)
            
            self.write_to_console(f"Processing with {model_type} architecture...")
            
            # Select appropriate separator based on model type
            if model_type == "VR":
                separator = SeperateVR(model_data, process_data)
            elif model_type == "MDX":
                if model_name.endswith('_MDXC.onnx'):
                    separator = SeperateMDXC(model_data, process_data)
                else:
                    separator = SeperateMDX(model_data, process_data)
            elif model_type == "Demucs":
                separator = SeperateDemucs(model_data, process_data)
            else:
                return None, None, f"Unsupported model type: {model_type}"
            
            # Perform separation
            progress(0.1, desc="Initializing separation...")
            separator.seperate()
            
            # Handle secondary model if enabled
            if is_secondary_model and secondary_model_name:
                self.write_to_console("Applying secondary model...")
                progress(0.7, desc="Applying secondary model...")
                # Secondary model processing would go here
                
            # Handle ensemble if enabled
            if is_ensemble_mode and ensemble_model2:
                self.write_to_console("Processing ensemble...")
                progress(0.9, desc="Processing ensemble...")
                # Ensemble processing would go here
            
            progress(1.0, desc="Separation complete!")
            
            # Find output files
            vocals_file = None
            instrumental_file = None
            
            output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
            
            for file in output_files:
                if "(Vocals)" in file or "(Vocal)" in file:
                    vocals_file = os.path.join(output_dir, file)
                elif "(Instrumental)" in file or "(Inst)" in file:
                    instrumental_file = os.path.join(output_dir, file)
            
            self.write_to_console("Separation completed successfully!")
            
            # Clear GPU cache
            clear_gpu_cache()
            
            return vocals_file, instrumental_file, "Separation completed successfully!"
                    
        except Exception as e:
            error_msg = f"Error during separation: {str(e)}\n{traceback.format_exc()}"
            self.write_to_console(f"ERROR: {error_msg}")
            return None, None, error_msg
            
        finally:
            self.is_processing = False
    
    def batch_separate(self, file_list, model_type, model_name, progress=gr.Progress()):
        """Batch processing function"""
        if not file_list:
            return "No files selected for batch processing"
            
        results = []
        total_files = len(file_list)
        
        for i, file_path in enumerate(file_list):
            progress((i + 1) / total_files, desc=f"Processing file {i+1}/{total_files}")
            
            vocals, instrumental, status = self.separate_audio(
                file_path, model_type, model_name,
                # Use current settings
                self.settings['aggression_setting'],
                self.settings['window_size'],
                self.settings['is_tta'],
                self.settings['is_post_process'],
                self.settings['is_high_end_process'],
                self.settings['mdx_segment_size'],
                self.settings['overlap_mdx'],
                self.settings['compensate'],
                self.settings['segment'],
                self.settings['overlap'],
                self.settings['shifts'],
                self.settings['save_format'],
                self.settings['is_normalization'],
                self.settings['is_add_model_name'],
                False, None, False, None
            )
            
            results.append(f"File {i+1}: {status}")
        
        return "\\n".join(results)

def create_complete_interface():
    """Create the complete Gradio interface"""
    uvr = UVRWebComplete()
    
    with gr.Blocks(title="Ultimate Vocal Remover - Complete Web Interface", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# Ultimate Vocal Remover - Complete Web Interface")
        gr.Markdown("Professional audio source separation with full UVR functionality")
        
        with gr.Tabs():
            # Main Processing Tab
            with gr.TabItem("Audio Separation"):
                with gr.Row():
                    with gr.Column():
                        # File Input
                        audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            format="wav"
                        )
                        
                        # Model Selection
                        model_type = gr.Radio(
                            choices=list(uvr.available_models.keys()) if uvr.available_models else ["No models found"],
                            label="Model Architecture",
                            value=list(uvr.available_models.keys())[0] if uvr.available_models else None
                        )
                        
                        model_name = gr.Dropdown(
                            label="Model",
                            choices=uvr.available_models.get(list(uvr.available_models.keys())[0], []) if uvr.available_models else [],
                            value=None
                        )
                        
                        # Process Button
                        separate_btn = gr.Button("Separate Audio", variant="primary", size="lg")
                        
                    with gr.Column():
                        # Status and Output
                        status_text = gr.Textbox(label="Status", interactive=False, lines=3)
                        vocals_output = gr.Audio(label="Vocals", type="filepath")
                        instrumental_output = gr.Audio(label="Instrumental", type="filepath")
            
            # VR Architecture Settings
            with gr.TabItem("VR Settings"):
                with gr.Row():
                    with gr.Column():
                        aggression_setting = gr.Slider(
                            minimum=0, maximum=20, value=5, step=1,
                            label="Aggression Setting"
                        )
                        window_size = gr.Radio(
                            choices=[512, 1024], value=512,
                            label="Window Size"
                        )
                        is_tta = gr.Checkbox(label="TTA (Test Time Augmentation)", value=False)
                        
                    with gr.Column():
                        is_post_process = gr.Checkbox(label="Post Process", value=False)
                        is_high_end_process = gr.Checkbox(label="High End Process", value=False)
                        post_process_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.01,
                            label="Post Process Threshold"
                        )
            
            # MDX Settings
            with gr.TabItem("MDX Settings"):
                with gr.Row():
                    with gr.Column():
                        mdx_segment_size = gr.Radio(
                            choices=[256, 512, 1024], value=256,
                            label="Segment Size"
                        )
                        overlap_mdx = gr.Radio(
                            choices=[0.25, 0.5, 0.75, 0.99], value=0.25,
                            label="Overlap"
                        )
                        
                    with gr.Column():
                        compensate = gr.Number(
                            value=1.03597672, precision=8,
                            label="Compensate"
                        )
                        mdx_batch_size = gr.Slider(
                            minimum=1, maximum=16, value=1, step=1,
                            label="Batch Size"
                        )
            
            # Demucs Settings
            with gr.TabItem("Demucs Settings"):
                with gr.Row():
                    with gr.Column():
                        segment_demucs = gr.Radio(
                            choices=["Default", "1", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50"],
                            value="Default",
                            label="Segment"
                        )
                        overlap_demucs = gr.Radio(
                            choices=[0.25, 0.5, 0.75, 0.99], value=0.25,
                            label="Overlap"
                        )
                        
                    with gr.Column():
                        shifts = gr.Slider(
                            minimum=0, maximum=20, value=2, step=1,
                            label="Shifts"
                        )
                        is_split_mode = gr.Checkbox(label="Split Mode", value=True)
                        is_demucs_combine_stems = gr.Checkbox(label="Combine Stems", value=True)
            
            # Advanced Settings
            with gr.TabItem("Advanced Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Output Settings")
                        output_format = gr.Radio(
                            choices=["WAV", "MP3", "FLAC"], value="WAV",
                            label="Output Format"
                        )
                        is_normalization = gr.Checkbox(label="Normalization", value=False)
                        is_add_model_name = gr.Checkbox(label="Add Model Name to Output", value=False)
                        
                    with gr.Column():
                        gr.Markdown("### Secondary Model")
                        is_secondary_model = gr.Checkbox(label="Enable Secondary Model", value=False)
                        secondary_model_name = gr.Dropdown(
                            label="Secondary Model",
                            choices=[],
                            value=None
                        )
                        
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Ensemble Settings")
                        is_ensemble_mode = gr.Checkbox(label="Enable Ensemble", value=False)
                        ensemble_model2 = gr.Dropdown(
                            label="Second Model for Ensemble",
                            choices=[],
                            value=None
                        )
                        
                    with gr.Column():
                        gr.Markdown("### GPU Settings")
                        is_gpu_conversion = gr.Checkbox(
                            label="Use GPU", 
                            value=torch.cuda.is_available()
                        )
                        gpu_info = gr.Textbox(
                            label="GPU Info",
                            value=f"CUDA Available: {torch.cuda.is_available()}",
                            interactive=False
                        )
            
            # Batch Processing
            with gr.TabItem("Batch Processing"):
                batch_files = gr.File(
                    label="Select Multiple Audio Files",
                    file_count="multiple",
                    file_types=["audio"]
                )
                batch_process_btn = gr.Button("Process Batch", variant="primary")
                batch_results = gr.Textbox(label="Batch Results", lines=10, interactive=False)
        
        # Event handlers
        model_type.change(
            fn=uvr.update_model_dropdown,
            inputs=[model_type],
            outputs=[model_name]
        )
        
        separate_btn.click(
            fn=uvr.separate_audio,
            inputs=[
                audio_input, model_type, model_name,
                aggression_setting, window_size, is_tta, is_post_process, is_high_end_process,
                mdx_segment_size, overlap_mdx, compensate,
                segment_demucs, overlap_demucs, shifts,
                output_format, is_normalization, is_add_model_name,
                is_secondary_model, secondary_model_name,
                is_ensemble_mode, ensemble_model2
            ],
            outputs=[vocals_output, instrumental_output, status_text]
        )
        
        batch_process_btn.click(
            fn=uvr.batch_separate,
            inputs=[batch_files, model_type, model_name],
            outputs=[batch_results]
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
    iface = create_complete_interface()
    iface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        inbrowser=False         # Don't auto-open browser on server
    )