"""
ComfyUI-LTX-GapFill
====================
Sends the last frame of clip A and the first frame of clip B to Google Gemini
and returns a suggested bridge prompt for LTX video generation.

Replicates the "Fill with Video" feature from LTX Desktop.
System prompt and API payload are verbatim from LTX Desktop source:
  backend/handlers/suggest_gap_prompt_handler.py @ Lightricks/LTX-Desktop
"""

from __future__ import annotations

import base64
import io
import json
import logging

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
TIMEOUT = 30


# ── Image helpers ─────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor, frame_index: int = 0) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 to PIL Image."""
    frame = tensor[frame_index]  # [H, W, C]
    arr = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resize_longest(pil_img: Image.Image, max_px: int) -> Image.Image:
    """Resize so the longest edge <= max_px, preserving aspect ratio."""
    w, h = pil_img.size
    longest = max(w, h)
    if longest <= max_px:
        return pil_img
    scale = max_px / longest
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _to_b64_jpeg(pil_img: Image.Image, quality: int = 85) -> str:
    """Encode a PIL image as base64 JPEG string."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_tensor_to_b64(
    tensor,
    pick_last: bool = False,
    resize: bool = True,
    max_px: int = 512,
) -> str:
    """
    Convert a ComfyUI IMAGE tensor to base64 JPEG.

    pick_last=True  → tensor[-1]  (last frame in batch  → use for clip BEFORE gap)
    pick_last=False → tensor[0]   (first frame in batch → use for clip AFTER gap)
    """
    idx = -1 if pick_last else 0
    pil = _tensor_to_pil(tensor, idx)
    if resize:
        pil = _resize_longest(pil, max_px)
    return _to_b64_jpeg(pil)


# ── Node ──────────────────────────────────────────────────────────────────────

class GeminiFillPrompt:
    """
    Gemini Fill Prompt (LTX Bridge)
    --------------------------------
    Replicates LTX Desktop's "Fill with Video" AI prompt suggestion.

    - last_frame  : last frame of the clip BEFORE the gap
                    (Load Image node  OR  VHS LoadVideo full batch — last frame auto-selected)
    - first_frame : first frame of the clip AFTER the gap
                    (Load Image node  OR  VHS LoadVideo full batch — first frame auto-selected)
    - prompt_before / prompt_after : paste the original clip prompts (optional)
    - gap_duration : length of the gap in seconds (type manually)
    - gemini_api_key : free key from https://aistudio.google.com/

    Returns a single STRING ready to wire into any LTX sampler / CLIP Text Encode node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Google Gemini API key. Free at https://aistudio.google.com/",
                    },
                ),
                "gap_duration": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.5,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": "Duration of the gap to fill, in seconds.",
                    },
                ),
                "resize_before_send": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Resize images before sending to Gemini. "
                            "Recommended — prevents hitting the 20 MB inline payload limit "
                            "and speeds up the request. Gemini only needs scene-level detail."
                        ),
                    },
                ),
                "max_size_px": (
                    "INT",
                    {
                        "default": 512,
                        "min": 128,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Longest edge in pixels when resize_before_send is enabled.",
                    },
                ),
            },
            "optional": {
                "last_frame": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Last frame of the clip BEFORE the gap. "
                            "Single image or full VHS video batch — last frame auto-selected."
                        ),
                    },
                ),
                "first_frame": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "First frame of the clip AFTER the gap. "
                            "Single image or full VHS video batch — first frame auto-selected."
                        ),
                    },
                ),
                "prompt_before": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt or description of the clip before the gap.",
                    },
                ),
                "prompt_after": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt or description of the clip after the gap.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("suggested_prompt",)
    FUNCTION = "suggest"
    CATEGORY = "LTX/Gap Fill"
    OUTPUT_NODE = False

    def suggest(
        self,
        gemini_api_key: str,
        gap_duration: float,
        resize_before_send: bool,
        max_size_px: int,
        last_frame=None,
        first_frame=None,
        prompt_before: str = "",
        prompt_after: str = "",
    ) -> tuple[str]:

        # ── Validation ────────────────────────────────────────────────────────
        if not gemini_api_key.strip():
            msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║         GeminiFillPrompt — API key not configured            ║\n"
                "║                                                              ║\n"
                "║  Get a FREE Gemini API key (no billing required):            ║\n"
                "║  https://aistudio.google.com/app/apikey                     ║\n"
                "║                                                              ║\n"
                "║  Paste the key into the gemini_api_key input on this node.  ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
            )
            print(msg)
            raise ValueError(
                "Gemini API key is required. "
                "Get one free at https://aistudio.google.com/app/apikey"
            )

        has_before = last_frame is not None or prompt_before.strip()
        has_after = first_frame is not None or prompt_after.strip()
        if not has_before and not has_after:
            raise ValueError(
                "At least one input is required: "
                "last_frame, first_frame, prompt_before, or prompt_after."
            )

        # ── System prompt (verbatim from LTX Desktop) ─────────────────────────
        system_text = (
            "You are a video production assistant. The user is editing a video timeline and has a gap "
            f"of {gap_duration:.1f} seconds between two shots. Your job is to suggest a detailed prompt "
            "for generating a video clip to fill this gap, so that it flows naturally between the "
            "preceding and following shots.\n\n"
            "Guidelines:\n"
            "- Describe the scene, action, camera movement, lighting, and mood\n"
            "- Match the visual style and tone of the surrounding shots\n"
            "- Create a smooth narrative or visual transition between the two shots\n"
            "- Keep the prompt concise (2-4 sentences max)\n"
            "- Write only the prompt text, no explanations or labels\n"
            "- If only one neighboring shot is available, suggest something that naturally "
            "leads into or out of it\n"
        )

        # ── Context text (verbatim structure from LTX Desktop) ────────────────
        context_text = "Here is the context from the timeline:\n\n"

        if has_before:
            context_text += "SHOT BEFORE THE GAP:\n"
            if prompt_before.strip():
                context_text += f"  Prompt: {prompt_before.strip()}\n"
            if last_frame is not None:
                context_text += "  Last frame (see image below):\n"

        if has_after:
            context_text += "\nSHOT AFTER THE GAP:\n"
            if prompt_after.strip():
                context_text += f"  Prompt: {prompt_after.strip()}\n"
            if first_frame is not None:
                context_text += "  First frame (see image below):\n"

        context_text += f"\nGap duration: {gap_duration:.1f} seconds\n"
        context_text += "Generation mode: text-to-video\n"
        context_text += "\nPlease suggest a detailed prompt for generating a video clip to fill this gap."

        # ── Assemble multipart user message ───────────────────────────────────
        user_parts: list[dict] = [{"text": context_text}]

        if last_frame is not None:
            b64 = _image_tensor_to_b64(
                last_frame,
                pick_last=True,
                resize=resize_before_send,
                max_px=max_size_px,
            )
            user_parts.append({"text": "Last frame of the shot BEFORE the gap:"})
            user_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})

        if first_frame is not None:
            b64 = _image_tensor_to_b64(
                first_frame,
                pick_last=False,
                resize=resize_before_send,
                max_px=max_size_px,
            )
            user_parts.append({"text": "First frame of the shot AFTER the gap:"})
            user_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})

        # ── Gemini payload ────────────────────────────────────────────────────
        payload = {
            "contents": [{"role": "user", "parts": user_parts}],
            "systemInstruction": {"parts": [{"text": system_text}]},
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 512,
            },
        }

        # ── HTTP request ──────────────────────────────────────────────────────
        headers = {
            "x-goog-api-key": gemini_api_key.strip(),
            "Content-Type": "application/json",
        }

        logger.info("[GeminiFillPrompt] Sending request to Gemini (gap=%.1fs)...", gap_duration)
        try:
            response = requests.post(
                GEMINI_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Gemini API timed out after {TIMEOUT}s. "
                "Check your connection or try reducing max_size_px."
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        if response.status_code == 400 and "API_KEY_INVALID" in response.text:
            msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║         GeminiFillPrompt — API key invalid                   ║\n"
                "║                                                              ║\n"
                "║  The key was rejected by Google. Generate a new one at:     ║\n"
                "║  https://aistudio.google.com/app/apikey                     ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
            )
            print(msg)
            raise RuntimeError(
                "Gemini API key is invalid. "
                "Generate a new one at https://aistudio.google.com/app/apikey"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API returned {response.status_code}: {response.text[:500]}"
            )

        # ── Parse response ────────────────────────────────────────────────────
        try:
            data = response.json()
            suggested_prompt = (
                data["candidates"][0]["content"]["parts"][0]["text"].strip()
            )
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to parse Gemini response: {exc}\nRaw: {response.text[:300]}"
            ) from exc

        logger.info("[GeminiFillPrompt] Suggested prompt: %s", suggested_prompt)
        print(f"\n[GeminiFillPrompt] Suggested bridge prompt:\n{suggested_prompt}\n")
        return (suggested_prompt,)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "GeminiFillPrompt": GeminiFillPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiFillPrompt": "Gemini Fill Prompt (LTX Bridge)",
}
