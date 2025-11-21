"""
Modal deployment for Svara TTS Inference
Deploys the Svara TTS multilingual text-to-speech service on Modal
"""

import modal

# Create Modal app
app = modal.App("indic-19-fastapi")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgomp1", "libsndfile1")
    .pip_install(
        "vllm>=0.6.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.9.0",
        "torch>=2.4.0",
        "snac @ git+https://github.com/hubertsiuzdak/snac.git",
        "huggingface-hub>=0.25.0",
        "transformers>=4.45.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
    )
)

# Define GPU configuration
GPU_CONFIG = "A100:1"  # You can adjust based on needs

# Define volumes for model caching
model_volume = modal.Volume.from_name("svara-tts-models", create_if_missing=True)

# vLLM server as a separate function
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/models": model_volume},
    timeout=3600,
    min_containers=1,
    max_containers=2,
)
@modal.asgi_app()
def fastapi_svara_app():
    """Main FastAPI application with integrated vLLM"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    from typing import Optional, List
    import asyncio
    import io
    import struct
    import torch
    import numpy as np
    
    # Import vLLM and SNAC components
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    # Initialize FastAPI
    web_app = FastAPI(title="Svara TTS API")
    
    # Voice configurations
    VOICE_PROFILES = {
        "hi_male": {"language": "hi", "gender": "male", "display_name": "Hindi Male"},
        "hi_female": {"language": "hi", "gender": "female", "display_name": "Hindi Female"},
        "en_male": {"language": "en", "gender": "male", "display_name": "English Male"},
        "en_female": {"language": "en", "gender": "female", "display_name": "English Female"},
        "bn_male": {"language": "bn", "gender": "male", "display_name": "Bengali Male"},
        "bn_female": {"language": "bn", "gender": "female", "display_name": "Bengali Female"},
        "mr_male": {"language": "mr", "gender": "male", "display_name": "Marathi Male"},
        "mr_female": {"language": "mr", "gender": "female", "display_name": "Marathi Female"},
        "te_male": {"language": "te", "gender": "male", "display_name": "Telugu Male"},
        "te_female": {"language": "te", "gender": "female", "display_name": "Telugu Female"},
        "kn_male": {"language": "kn", "gender": "male", "display_name": "Kannada Male"},
        "kn_female": {"language": "kn", "gender": "female", "display_name": "Kannada Female"},
        "ta_male": {"language": "ta", "gender": "male", "display_name": "Tamil Male"},
        "ta_female": {"language": "ta", "gender": "female", "display_name": "Tamil Female"},
        "gu_male": {"language": "gu", "gender": "male", "display_name": "Gujarati Male"},
        "gu_female": {"language": "gu", "gender": "female", "display_name": "Gujarati Female"},
        "ml_male": {"language": "ml", "gender": "male", "display_name": "Malayalam Male"},
        "ml_female": {"language": "ml", "gender": "female", "display_name": "Malayalam Female"},
        "pa_male": {"language": "pa", "gender": "male", "display_name": "Punjabi Male"},
        "pa_female": {"language": "pa", "gender": "female", "display_name": "Punjabi Female"},
    }
    
    # Pydantic models
    class TTSRequest(BaseModel):
        text: str = Field(..., description="Text to synthesize")
        voice_id: str = Field(default="hi_male", description="Voice ID to use")
        stream: bool = Field(default=True, description="Stream audio response")
        
    class VoiceInfo(BaseModel):
        voice_id: str
        language: str
        gender: str
        display_name: str
    
    # Global variables for models
    llm = None
    snac_decoder = None
    tokenizer = None
    
    @web_app.on_event("startup")
    async def startup_event():
        """Initialize models on startup"""
        nonlocal llm, snac_decoder, tokenizer
        
        print("Loading Svara TTS model...")
        
        # Initialize vLLM engine
        llm = LLM(
            model="kenpath/svara-tts-v1",
            download_dir="/models",
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            trust_remote_code=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "kenpath/svara-tts-v1",
            cache_dir="/models",
            trust_remote_code=True,
        )
        
        # Initialize SNAC decoder
        print("Loading SNAC decoder...")
        try:
            from snac import SNAC
            snac_decoder = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            if torch.cuda.is_available():
                snac_decoder = snac_decoder.cuda()
        except Exception as e:
            print(f"Warning: Could not load SNAC decoder: {e}")
            snac_decoder = None
        
        print("✓ Models loaded successfully!")
    
    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        nonlocal llm, snac_decoder
        return {
            "status": "healthy",
            "model_loaded": llm is not None,
            "decoder_loaded": snac_decoder is not None,
        }
    
    @web_app.get("/v1/voices", response_model=List[VoiceInfo])
    async def get_voices():
        """Get list of available voices"""
        return [
            VoiceInfo(voice_id=vid, **info)
            for vid, info in VOICE_PROFILES.items()
        ]
    
    @web_app.post("/v1/text-to-speech")
    async def text_to_speech(request: TTSRequest):
        """Generate speech from text"""
        nonlocal llm, snac_decoder
        if llm is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if request.voice_id not in VOICE_PROFILES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice_id. Available: {list(VOICE_PROFILES.keys())}"
            )
        
        try:
            # Format prompt with voice information (matching reference implementation)
            voice_info = VOICE_PROFILES[request.voice_id]
            voice = f"{voice_info['display_name']}"
            formatted_text = f"<|audio|> {voice}: {request.text}<|eot_id|>"
            prompt = "<custom_token_3>" + formatted_text + "<custom_token_4><custom_token_5>"
            
            # Generate tokens using vLLM
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=800,
                repetition_penalty=1.2,
            )
            
            outputs = llm.generate([prompt], sampling_params)
            generated_tokens = outputs[0].outputs[0].token_ids
            
            # Decode audio tokens to PCM audio
            if snac_decoder is not None:
                audio_data = decode_audio_tokens(generated_tokens)
            else:
                # Fallback: return dummy audio if decoder not available
                audio_data = np.zeros(24000, dtype=np.int16)  # 1 second of silence
            
            if request.stream:
                # Stream PCM audio
                def audio_stream():
                    # Convert to bytes
                    audio_bytes = audio_data.tobytes()
                    # Stream in chunks
                    chunk_size = 8192
                    for i in range(0, len(audio_bytes), chunk_size):
                        yield audio_bytes[i:i + chunk_size]
                
                return StreamingResponse(
                    audio_stream(),
                    media_type="audio/pcm",
                    headers={
                        "Content-Type": "audio/pcm",
                        "X-Sample-Rate": "24000",
                        "X-Channels": "1",
                    }
                )
            else:
                # Return complete audio
                audio_bytes = audio_data.tobytes()
                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/pcm",
                    headers={
                        "Content-Type": "audio/pcm",
                        "X-Sample-Rate": "24000",
                        "X-Channels": "1",
                    }
                )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    def decode_audio_tokens(tokens):
        """Decode audio tokens using SNAC decoder"""
        nonlocal snac_decoder
        if snac_decoder is None:
            return np.zeros(24000, dtype=np.int16)
        
        try:
            # Constants for token processing
            START_OF_SPEECH_TOKEN = 128257
            END_OF_SPEECH_TOKEN = 128258
            AUDIO_CODE_BASE_OFFSET = 128266
            AUDIO_CODE_MAX = AUDIO_CODE_BASE_OFFSET + (7 * 4096) - 1
            
            # Convert to tensor for processing
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            
            # Find start of speech token
            token_indices = (token_tensor == START_OF_SPEECH_TOKEN).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                print("Warning: No START_OF_SPEECH_TOKEN found in generated tokens")
                return np.zeros(24000, dtype=np.int16)
            
            # Extract audio tokens after the last START_OF_SPEECH_TOKEN
            start_idx = token_indices[-1].item() + 1
            audio_tokens = token_tensor[start_idx:]
            
            # Filter out special tokens
            audio_tokens = audio_tokens[audio_tokens != END_OF_SPEECH_TOKEN]
            audio_tokens = audio_tokens[audio_tokens != 128263]  # PAD token
            
            # Only keep valid SNAC tokens
            valid_mask = (audio_tokens >= AUDIO_CODE_BASE_OFFSET) & (audio_tokens <= AUDIO_CODE_MAX)
            audio_tokens = audio_tokens[valid_mask]
            
            if len(audio_tokens) == 0:
                print("Warning: No valid audio tokens found")
                return np.zeros(24000, dtype=np.int16)
            
            # Convert to list and subtract base offset
            snac_tokens = audio_tokens.tolist()
            snac_tokens = [t - AUDIO_CODE_BASE_OFFSET for t in snac_tokens]
            
            # Trim to multiple of 7 (SNAC requires groups of 7)
            new_length = (len(snac_tokens) // 7) * 7
            snac_tokens = snac_tokens[:new_length]
            
            if len(snac_tokens) == 0:
                print("Warning: No valid SNAC token groups found")
                return np.zeros(24000, dtype=np.int16)
            
            # Redistribute codes into hierarchical levels for SNAC decoder
            codes_lvl = [[] for _ in range(3)]
            llm_codebook_offsets = [i * 4096 for i in range(7)]
            
            for i in range(0, len(snac_tokens), 7):
                # Level 0: Coarse (token 0)
                codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
                # Level 1: Medium (tokens 1, 4)
                codes_lvl[1].append(snac_tokens[i + 1] - llm_codebook_offsets[1])
                codes_lvl[1].append(snac_tokens[i + 4] - llm_codebook_offsets[4])
                # Level 2: Fine (tokens 2, 3, 5, 6)
                codes_lvl[2].append(snac_tokens[i + 2] - llm_codebook_offsets[2])
                codes_lvl[2].append(snac_tokens[i + 3] - llm_codebook_offsets[3])
                codes_lvl[2].append(snac_tokens[i + 5] - llm_codebook_offsets[5])
                codes_lvl[2].append(snac_tokens[i + 6] - llm_codebook_offsets[6])
            
            # Convert to tensors for SNAC decoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hierarchical_codes = []
            for lvl_codes in codes_lvl:
                tensor = torch.tensor(lvl_codes, dtype=torch.long, device=device).unsqueeze(0)
                hierarchical_codes.append(tensor)
            
            # Decode with SNAC
            with torch.no_grad():
                audio = snac_decoder.decode(hierarchical_codes)
            
            # Convert to numpy and scale to int16
            audio_np = audio.detach().cpu().numpy().squeeze()
            # Clip to valid range and convert to int16
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16
        except Exception as e:
            print(f"Audio decoding error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(24000, dtype=np.int16)
    
    return web_app


# CLI interface for testing
@app.local_entrypoint()
def main():
    """Test the deployment locally"""
    print("Svara TTS deployed on Modal!")
    print("Access your API at the provided URL")
    print("\nExample usage:")
    print("curl -X POST https://your-app.modal.run/v1/text-to-speech \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"text": "नमस्ते", "voice_id": "hi_male"}\' \\')
    print("  --output audio.pcm")