from __future__ import annotations

"""
Modal app that wraps the svara-tts-v1 notebook logic in a FastAPI server.

Deployment:
    modal deploy modal_app.py
"""

import modal
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Modal App & Image
# -----------------------------------------------------------------------------

app = modal.App("indic-19")  # name can be whatever you like

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install runtime deps (mirrors your stack)
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "torchaudio>=2.5.0",
        "transformers>=4.57.0",
        "accelerate>=1.10.0",
        "vllm>=0.11.0",
        "xformers>=0.0.32",
        "snac>=1.2.1",
        "soundfile>=0.13.0",
        "numpy>=2.1.0",
        "librosa>=0.11.0",
        "scipy>=1.15.0",
        "fastapi>=0.119.0",
        "uvicorn[standard]>=0.38.0",
        "pydantic>=2.12.0",
        "pydantic-settings>=2.11.0",
        "python-multipart>=0.0.20",
        "httpx>=0.28.0",
        "python-dotenv>=1.1.0",
        "huggingface-hub>=0.35.0",
        "tqdm>=4.67.0",
        "openai>=2.5.0",
        "python-Levenshtein>=0.21.0",
    )
)


# -----------------------------------------------------------------------------
# Model Loading (same logic as svara_TTS_Inference.ipynb), but lazy imports
# -----------------------------------------------------------------------------

MODEL_NAME = "kenpath/svara-tts-v1"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000

_device = None
_snac_model = None
_lm_model = None
_tokenizer = None


def _get_device():
    """Lazily determine the best available device (prefers GPU)."""
    global _device
    import torch  # heavy import only when actually needed

    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def _load_models():
    """
    Lazily import and load the SNAC decoder and the Svara TTS model.

    Mirrors initialization logic in svara_TTS_Inference.ipynb,
    but defers heavy imports to runtime.
    """
    global _snac_model, _lm_model, _tokenizer

    # heavy imports live here instead of top-level
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _get_device()

    if _snac_model is None:
        _snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).to(device)

    if _lm_model is None or _tokenizer is None:
        _lm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return _snac_model, _lm_model, _tokenizer, device


def generate_audio_from_text(text: str, language: str, gender: str) -> np.ndarray:
    """
    Generate audio from text using the Svara-TTS model.

    Args:
        text: The text to synthesize into speech.
        language: The language name (e.g., 'Hindi', 'Bengali', 'English').
        gender: The gender of the voice ('Male' or 'Female').

    Returns:
        Numpy array of audio samples at SAMPLE_RATE (24kHz).
    """
    import torch  # local import, cached after first time

    snac_model, model, tokenizer, device = _load_models()

    # ----- Prompt formatting (same as notebook) -----
    voice = f"{language} ({gender})"
    formatted_text = f"<|audio|> {voice}: {text}<|eot_id|>"
    prompt = "<custom_token_3>" + formatted_text + "<custom_token_4><custom_token_5>"

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Add special tokens
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    # Move to device
    input_ids = modified_input_ids.to(device)

    # ----- Generate speech tokens (same hyperparameters as notebook) -----
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=1,
            eos_token_id=128258,
        )

    # ----- Parse output tokens to extract SNAC codes (same as notebook) -----
    START_OF_SPEECH_TOKEN = 128257
    END_OF_SPEECH_TOKEN = 128258
    AUDIO_CODE_BASE_OFFSET = 128266
    AUDIO_CODE_MAX = AUDIO_CODE_BASE_OFFSET + (7 * 4096) - 1

    row = generated_ids[0]
    token_indices = (row == START_OF_SPEECH_TOKEN).nonzero(as_tuple=True)[0]

    if len(token_indices) > 0:
        start_idx = token_indices[-1].item() + 1
        audio_tokens = row[start_idx:]
        audio_tokens = audio_tokens[audio_tokens != END_OF_SPEECH_TOKEN]
        audio_tokens = audio_tokens[audio_tokens != 128263]  # PAD token

        # Only keep valid SNAC tokens
        valid_mask = (audio_tokens >= AUDIO_CODE_BASE_OFFSET) & (
            audio_tokens <= AUDIO_CODE_MAX
        )
        audio_tokens = audio_tokens[valid_mask]

        snac_tokens = audio_tokens.tolist()
        snac_tokens = [t - AUDIO_CODE_BASE_OFFSET for t in snac_tokens]

        # Trim to multiple of 7
        new_length = (len(snac_tokens) // 7) * 7
        snac_tokens = snac_tokens[:new_length]
    else:
        raise ValueError("No speech tokens found in generated output")

    # Redistribute codes into hierarchical levels for SNAC decoder
    def redistribute_codes(code_list):
        """De-interleave SNAC tokens into 3 hierarchical levels."""
        import torch  # ensure available in nested scope (still cached)

        codes_lvl = [[] for _ in range(3)]
        llm_codebook_offsets = [i * 4096 for i in range(7)]

        for i in range(0, len(code_list), 7):
            # Level 0: Coarse
            codes_lvl[0].append(code_list[i] - llm_codebook_offsets[0])
            # Level 1: Medium
            codes_lvl[1].append(code_list[i + 1] - llm_codebook_offsets[1])
            codes_lvl[1].append(code_list[i + 4] - llm_codebook_offsets[4])
            # Level 2: Fine
            codes_lvl[2].append(code_list[i + 2] - llm_codebook_offsets[2])
            codes_lvl[2].append(code_list[i + 3] - llm_codebook_offsets[3])
            codes_lvl[2].append(code_list[i + 5] - llm_codebook_offsets[5])
            codes_lvl[2].append(code_list[i + 6] - llm_codebook_offsets[6])

        # Convert to tensors for SNAC decoder
        hierarchical_codes = []
        for lvl_codes in codes_lvl:
            tensor = torch.tensor(
                lvl_codes, dtype=torch.long, device=device
            ).unsqueeze(0)
            hierarchical_codes.append(tensor)

        # Decode with SNAC
        with torch.no_grad():
            audio_hat = snac_model.decode(hierarchical_codes)

        return audio_hat

    # Generate audio waveform
    audio_waveform = redistribute_codes(snac_tokens)

    # Convert to numpy array
    audio_array = audio_waveform.detach().squeeze().to("cpu").numpy()

    return audio_array


def audio_array_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """
    Convert float audio in [-1, 1] to 16-bit PCM little-endian bytes.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * 32767.0).astype(np.int16)
    return int16.tobytes()


# -----------------------------------------------------------------------------
# FastAPI schema & app
# -----------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str
    language: str
    gender: str


def create_fastapi_app() -> FastAPI:
    api = FastAPI(
        title="Svara TTS (Modal)",
        description="FastAPI wrapper around svara-tts-v1 using the notebook logic.",
        version="1.0.0",
    )

    @api.get("/health")
    async def health():
        # Trigger lazy load once to verify model availability
        _load_models()
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "snac_model": SNAC_MODEL_NAME,
            "sample_rate": SAMPLE_RATE,
        }

    @api.post("/v1/text-to-speech", response_class=Response)
    async def text_to_speech(req: TTSRequest):
        """
        Generate speech audio from text.

        Request body:
        {
            "text": "...",
            "language": "Hindi",
            "gender": "Female"
        }

        Response:
            Raw 16-bit PCM audio at 24kHz with content-type `audio/pcm`.
        """
        audio_array = generate_audio_from_text(
            text=req.text, language=req.language, gender=req.gender
        )
        pcm_bytes = audio_array_to_pcm16_bytes(audio_array)
        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Bit-Depth": "16",
                "X-Channels": "1",
            },
        )

    return api


# -----------------------------------------------------------------------------
# Modal ASGI entrypoint (matches the pattern you showed)
# -----------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100:1",
    min_containers=1,
    max_containers=2,
    timeout=10 * 60,
)
@modal.asgi_app()
def tts_app():
    """
    Runs inside the Modal container.

    Returns the FastAPI app as the ASGI app.
    """
    return create_fastapi_app()


if __name__ == "__main__":
    # Optional: local dev server (without Modal) if you have deps installed.
    import uvicorn

    uvicorn.run(
        create_fastapi_app(),
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
