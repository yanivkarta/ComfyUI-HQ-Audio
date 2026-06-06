import os
import torch
import soundfile as sf
import numpy as np
import folder_paths
from server import PromptServer

class HighQualityAudioSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "hq_audio"}),
                "format": (["WAV", "AIFF"], {"default": "WAV"}),
                "sample_rate": (["Input Native", 44100, 48000, 88200, 96000, 192000], {"default": "192000"}),
                "bit_depth": (["16-bit", "24-bit", "32-bit float"], {"default": "24-bit"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save_audio(self, audio, filename_prefix="hq_audio", format="WAV", sample_rate="192000", bit_depth="24-bit"):
        subtype_map = {
            "16-bit": "PCM_16",
            "24-bit": "PCM_24",
            "32-bit float": "FLOAT"
        }
        subtype = subtype_map.get(bit_depth, "PCM_24")

        waveform = audio.get("waveform")
        incoming_sample_rate = audio.get("sample_rate", 44100)

        if waveform is None:
            self._send_error("No valid waveform matrix discovered.")
            return {"ui": {"audio": []}}

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        # 1. Standardize dimensions to [Batch, Channels, Samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() != 3:
            self._send_error(f"Unsupported tensor shape: {list(waveform.shape)}.")
            return {"ui": {"audio": []}}

        if waveform.shape[1] > waveform.shape[2]:
            waveform = waveform.transpose(1, 2)

        # 2. Determine target sample rate and handle dynamic resampling
        if sample_rate == "Input Native":
            target_sample_rate = incoming_sample_rate
        else:
            target_sample_rate = int(sample_rate)

        if incoming_sample_rate != target_sample_rate:
            print(f"[HQ Audio Saver] Resampling audio from {incoming_sample_rate}Hz to {target_sample_rate}Hz...")
            old_samples = waveform.shape[2]
            new_samples = int(old_samples * (target_sample_rate / incoming_sample_rate))
            
            # High-fidelity linear interpolation across the sample dimension using PyTorch
            waveform = torch.nn.functional.interpolate(
                waveform, 
                size=new_samples, 
                mode='linear', 
                align_corners=False
            )

        batch_size = waveform.shape[0]
        results = []

        for i in range(batch_size):
            try:
                self._send_progress(i, batch_size)
                
                # Convert to NumPy and transpose to [Samples, Channels] for libsndfile
                audio_np = waveform[i].detach().cpu().numpy().T 

                full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                    filename_prefix, self.output_dir, waveform[i].shape[1], waveform[i].shape[0]
                )
                
                file_ext = format.lower()
                file_name_with_ext = f"{filename}_{counter:05d}.{file_ext}"
                file_path = os.path.join(full_output_folder, file_name_with_ext)

                # Write to disk using optimized C-backend
                sf.write(file_path, audio_np, target_sample_rate, subtype=subtype, format=format)

                results.append({
                    "filename": file_name_with_ext,
                    "subfolder": subfolder,
                    "type": self.type
                })
                
                print(f"[HQ Audio Saver] Saved: {file_path} ({bit_depth} @ {target_sample_rate}Hz, Duration: {audio_np.shape[0]/target_sample_rate:.2f}s)")

            except Exception as e:
                self._send_error(f"Save error: {str(e)}")

        self._send_progress(batch_size, batch_size)
        return {"ui": {"audio": results}}

    def _send_progress(self, current, total):
        PromptServer.instance.send_sync("progress", {"value": current, "max": total})

    def _send_error(self, message):
        print(f"[HQ Audio Saver Error] {message}")
        PromptServer.instance.send_sync("execution_error", {"message": message})


NODE_CLASS_MAPPINGS = {
    "HighQualityAudioSaver": HighQualityAudioSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighQualityAudioSaver": "Save Audio (High Quality)"
}
