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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
TIMEOUT = 30

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

PROMPT_STYLES = [
    "LTX Desktop system prompt",  # Verbatim LTX Desktop — 2-4 sentences
    "detailed",                   # Full LTX-style prompt — camera, motion, lighting, atmosphere
    "cinematic",                  # Emphasis on cinematography and visual language
    "narrative",                  # Story-driven — character, action, emotional arc
]

# Guidelines block swapped into system prompt based on selected style
_STYLE_GUIDELINES: dict[str, str] = {
    "LTX Desktop system prompt": (
        "- Describe the scene, action, camera movement, lighting, and mood\n"
        "- Match the visual style and tone of the surrounding shots\n"
        "- Create a smooth narrative or visual transition between the two shots\n"
        "- Keep the prompt concise (2-4 sentences max)\n"
        "- Write only the prompt text, no explanations or labels\n"
        "- If only one neighboring shot is available, suggest something that naturally "
        "leads into or out of it\n"
    ),
    "detailed": (
        "- Write a detailed LTX video generation prompt (6-10 sentences)\n"
        "- Describe subject action and movement with specific direction and speed\n"
        "- Include camera motion: pan, tilt, dolly, zoom, handheld, or static\n"
        "- Describe lighting quality, color temperature, and shadows\n"
        "- Include color palette, saturation, and overall visual tone\n"
        "- Describe depth of field, focus plane, and background treatment\n"
        "- Note atmosphere: time of day, weather, environmental mood\n"
        "- Match the visual style and continuity of the surrounding shots\n"
        "- Write only the prompt text, no explanations or labels\n"
        "- If only one neighboring shot is available, suggest something that naturally "
        "leads into or out of it\n"
    ),
    "cinematic": (
        "- Write a cinematic video generation prompt (5-8 sentences)\n"
        "- Lead with the camera setup: lens feel, framing, and movement\n"
        "- Describe the shot type: wide, medium, close-up, extreme close-up, POV, etc.\n"
        "- Specify camera motion precisely: slow push-in, whip pan, crane up, tracking shot, etc.\n"
        "- Describe the lighting like a cinematographer: key light direction, fill ratio, "
        "practical lights, motivated sources\n"
        "- Include film stock feel, grain, color grading style, or visual reference if relevant\n"
        "- Match the cinematic language and visual grammar of the surrounding shots\n"
        "- Write only the prompt text, no explanations or labels\n"
        "- If only one neighboring shot is available, suggest something that naturally "
        "leads into or out of it\n"
    ),
    "narrative": (
        "- Write a narrative-driven video generation prompt (5-8 sentences)\n"
        "- Focus on the emotional arc and story beat this clip represents\n"
        "- Describe character action, expression, and motivation where applicable\n"
        "- Include what changes or is revealed during this clip\n"
        "- Ground the scene in specific sensory detail: sound, texture, temperature, smell\n"
        "- Describe how the visual tone supports the emotional content\n"
        "- Ensure the clip advances or connects the story of the surrounding shots\n"
        "- Write only the prompt text, no explanations or labels\n"
        "- If only one neighboring shot is available, suggest something that naturally "
        "leads into or out of it\n"
    ),
}


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

    - clip_before_last_frame  : connect the clip BEFORE the gap — last frame auto-selected
                                (Load Image node  OR  VHS LoadVideo full batch)
    - clip_after_first_frame  : connect the clip AFTER the gap — first frame auto-selected
                                (Load Image node  OR  VHS LoadVideo full batch)
    - prompt_before / prompt_after : paste the original clip prompts (optional)
    - gap_duration : length of the gap in seconds (type manually)
    - gemini_api_key : free key from https://aistudio.google.com/

    Returns a single STRING ready to wire into any LTX sampler / CLIP Text Encode node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": (
                            "ComfyUI workflow helper only — this value is NOT sent to Gemini. "
                            "Gemini has no seed concept. "
                            "Change this value to force the node to re-run when all other inputs are unchanged."
                        ),
                    },
                ),
                "gemini_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Google Gemini API key. Free at https://aistudio.google.com/",
                    },
                ),
                "model": (
                    GEMINI_MODELS,
                    {
                        "default": "gemini-2.5-flash",
                        "tooltip": "Gemini model to use. All listed models are available on the free tier with limitations.",
                    },
                ),
                "custom_model": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Override the model dropdown with any Gemini model ID (e.g. gemini-3.2-flash). Leave blank to use the dropdown selection.",
                    },
                ),
                "prompt_style": (
                    PROMPT_STYLES,
                    {
                        "default": "LTX Desktop system prompt",
                        "tooltip": (
                            "LTX Desktop system prompt — verbatim LTX Desktop prompt, 2-4 sentences.\n"
                            "detailed  — 6-10 sentences, full camera/lighting/motion description.\n"
                            "cinematic — 5-8 sentences, cinematographer framing and lighting language.\n"
                            "narrative — 5-8 sentences, story-driven with emotional arc and character action."
                        ),
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
                "clip_before_last_frame": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Connect the clip BEFORE the gap here. "
                            "The node automatically uses the last frame. "
                            "Single image or full VHS video batch — last frame auto-selected."
                        ),
                    },
                ),
                "clip_after_first_frame": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Connect the clip AFTER the gap here. "
                            "The node automatically uses the first frame. "
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
                "custom_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "WARNING: Overrides the prompt_style dropdown entirely with your own system prompt.\n"
                            "Use with caution — a poorly written system prompt can produce vague or unusable results, "
                            "wasting API tokens and counting against your daily quota.\n"
                            "Leave blank to use the prompt_style dropdown (recommended)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("suggested_prompt", "report")
    FUNCTION = "suggest"
    CATEGORY = "LTX/Gap Fill"
    OUTPUT_NODE = False

    def suggest(
        self,
        seed: int,
        gemini_api_key: str,
        model: str,
        custom_model: str,
        prompt_style: str,
        gap_duration: float,
        resize_before_send: bool,
        max_size_px: int,
        clip_before_last_frame=None,
        clip_after_first_frame=None,
        prompt_before: str = "",
        prompt_after: str = "",
        custom_system_prompt: str = "",
    ) -> tuple[str]:

        active_model = custom_model.strip() if custom_model.strip() else model

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

        has_before = clip_before_last_frame is not None or prompt_before.strip()
        has_after = clip_after_first_frame is not None or prompt_after.strip()
        if not has_before and not has_after:
            raise ValueError(
                "At least one input is required: "
                "clip_before_last_frame, clip_after_first_frame, prompt_before, or prompt_after."
            )

        # ── System prompt ─────────────────────────────────────────────────────
        using_custom = bool(custom_system_prompt.strip())

        if using_custom:
            system_text = custom_system_prompt.strip()
            max_tokens  = 1024
            print(
                "\n[GeminiFillPrompt] ⚠  WARNING: custom_system_prompt is active — "
                "prompt_style dropdown is ignored.\n"
                "[GeminiFillPrompt]    A poorly written system prompt wastes tokens and "
                "counts against your daily quota.\n"
            )
        else:
            # Base is verbatim from LTX Desktop; guidelines block swapped by style.
            guidelines = _STYLE_GUIDELINES.get(prompt_style, _STYLE_GUIDELINES["LTX Desktop system prompt"])
            max_tokens  = 256 if prompt_style == "LTX Desktop system prompt" else 1024
            system_text = (
                "You are a video production assistant. The user is editing a video timeline and has a gap "
                f"of {gap_duration:.1f} seconds between two shots. Your job is to suggest a detailed prompt "
                "for generating a video clip to fill this gap, so that it flows naturally between the "
                "preceding and following shots.\n\n"
                "Guidelines:\n"
                + guidelines
            )

        # ── Context text (verbatim structure from LTX Desktop) ────────────────
        context_text = "Here is the context from the timeline:\n\n"

        if has_before:
            context_text += "SHOT BEFORE THE GAP:\n"
            if prompt_before.strip():
                context_text += f"  Prompt: {prompt_before.strip()}\n"
            if clip_before_last_frame is not None:
                context_text += "  Last frame (see image below):\n"

        if has_after:
            context_text += "\nSHOT AFTER THE GAP:\n"
            if prompt_after.strip():
                context_text += f"  Prompt: {prompt_after.strip()}\n"
            if clip_after_first_frame is not None:
                context_text += "  First frame (see image below):\n"

        context_text += f"\nGap duration: {gap_duration:.1f} seconds\n"
        context_text += "Generation mode: text-to-video\n"
        context_text += "\nPlease suggest a detailed prompt for generating a video clip to fill this gap."

        # ── Assemble multipart user message ───────────────────────────────────
        user_parts: list[dict] = [{"text": context_text}]

        if clip_before_last_frame is not None:
            b64 = _image_tensor_to_b64(
                clip_before_last_frame,
                pick_last=True,
                resize=resize_before_send,
                max_px=max_size_px,
            )
            user_parts.append({"text": "Last frame of the shot BEFORE the gap:"})
            user_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})

        if clip_after_first_frame is not None:
            b64 = _image_tensor_to_b64(
                clip_after_first_frame,
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
                "maxOutputTokens": max_tokens,
            },
        }

        # ── HTTP request ──────────────────────────────────────────────────────
        headers = {
            "x-goog-api-key": gemini_api_key.strip(),
            "Content-Type": "application/json",
        }

        endpoint = f"{GEMINI_BASE_URL}/{active_model}:generateContent"
        print(f"[GeminiFillPrompt] Seed             : {seed} (ComfyUI cache-buster only — not sent to Gemini)")
        print(f"[GeminiFillPrompt] Contacting model : {active_model}")
        print(f"[GeminiFillPrompt] Endpoint         : {endpoint}")
        if using_custom:
            print(f"[GeminiFillPrompt] Prompt style     : CUSTOM SYSTEM PROMPT  (max_tokens={max_tokens})")
        else:
            print(f"[GeminiFillPrompt] Prompt style     : {prompt_style}  (max_tokens={max_tokens})")
        logger.info("[GeminiFillPrompt] Sending request to %s (gap=%.1fs)...", active_model, gap_duration)
        try:
            response = requests.post(
                endpoint,
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

        if response.status_code == 429:
            msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║         GeminiFillPrompt — Quota exceeded (429)              ║\n"
                "║                                                              ║\n"
                f"║  Model : {active_model:<52}║\n"
                "║                                                              ║\n"
                "║  Your free tier quota for this model has been exhausted.    ║\n"
                "║  Options:                                                    ║\n"
                "║    1. Wait until the quota resets (daily, ~midnight PT)     ║\n"
                "║    2. Enable billing to remove the daily cap:               ║\n"
                "║       https://console.cloud.google.com/billing              ║\n"
                "║    3. Monitor your current rate limit usage:                ║\n"
                "║       https://ai.dev/rate-limit                             ║\n"
                "║    4. Review Gemini API rate limit docs:                    ║\n"
                "║       https://ai.google.dev/gemini-api/docs/rate-limits     ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
            )
            print(msg)
            raise RuntimeError(
                f"Gemini quota exceeded for model '{active_model}'. "
                "Check https://ai.dev/rate-limit to monitor usage, "
                "or enable billing at https://console.cloud.google.com/billing"
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

        # ── Build report ──────────────────────────────────────────────────────
        W    = 64
        div  = "╠" + "═" * W + "╣"

        def _row(label: str, value: str) -> str:
            content = f"{label:<16} {value}"
            if len(content) > W - 2:
                content = content[:W - 5] + "..."
            return f"║  {content:<{W - 2}}║"

        def _section(title: str) -> str:
            padding = W - len(title) - 4
            return f"║  [{title}]{' ' * max(padding, 0)}║"

        def _wrap(text: str) -> list[str]:
            out = []
            for raw in text.splitlines():
                raw = raw or ""
                while len(raw) > W - 4:
                    out.append(f"║  {raw[:W - 4]:<{W - 2}}║")
                    raw = "    " + raw[W - 4:]
                out.append(f"║  {raw:<{W - 2}}║")
            return out

        def _frame_info(tensor, pick_last: bool) -> str:
            if tensor is None:
                return "None"
            pil = _tensor_to_pil(tensor, -1 if pick_last else 0)
            orig_w, orig_h = pil.size
            if resize_before_send:
                pil = _resize_longest(pil, max_size_px)
            send_w, send_h = pil.size
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="JPEG", quality=85)
            kb = len(buf.getvalue()) / 1024
            return f"{orig_w}x{orig_h} → sends {send_w}x{send_h} JPEG ~{kb:.0f} KB"

        style_label = "⚠ CUSTOM SYSTEM PROMPT" if using_custom else prompt_style
        est_kb  = sum(
            len((lambda p: (p.convert("RGB"), io.BytesIO())[1])(
                _resize_longest(_tensor_to_pil(t, -1 if pl else 0), max_size_px)
                if resize_before_send else _tensor_to_pil(t, -1 if pl else 0)
            ).getvalue()) * 1.33 / 1024
            for t, pl in [(clip_before_last_frame, True), (clip_after_first_frame, False)]
            if t is not None
        ) + (len(prompt_before) + len(prompt_after) + len(system_text) + 800) / 1024
        est_str = f"~{est_kb:.0f} KB" if est_kb < 1024 else f"~{est_kb / 1024:.1f} MB"

        report_lines = [
            "",
            "╔" + "═" * W + "╗",
            f"║{'  LTX-GapFill Prompt — Run Report':^{W}}║",
            div,
            _row("Model",     active_model),
            _row("Endpoint",  endpoint),
            _row("Style",     f"{style_label}  (max_tokens={max_tokens})"),
            _row("Gap",       f"{gap_duration:.1f} seconds"),
            _row("Resize",    f"{'Yes' if resize_before_send else 'No'} (max {max_size_px}px)"),
            _row("Est. size", est_str),
            div,
            _section("SYSTEM PROMPT SENT TO GEMINI"),
        ]
        report_lines += _wrap(system_text)
        report_lines += [
            div,
            _section("SHOT BEFORE (clip_before_last_frame)"),
            _row("Frame",  _frame_info(clip_before_last_frame, True)),
            _row("Prompt", (prompt_before.strip()[:W - 20] + "...") if len(prompt_before.strip()) > W - 20 else (prompt_before.strip() or "(none)")),
            div,
            _section("SHOT AFTER (clip_after_first_frame)"),
            _row("Frame",  _frame_info(clip_after_first_frame, False)),
            _row("Prompt", (prompt_after.strip()[:W - 20] + "...") if len(prompt_after.strip()) > W - 20 else (prompt_after.strip() or "(none)")),
            div,
            _section("GEMINI RESPONSE"),
        ]
        report_lines += _wrap(suggested_prompt)
        report_lines += [
            "╚" + "═" * W + "╝",
            "",
        ]
        report = "\n".join(report_lines)

        print(f"\n[GeminiFillPrompt] Suggested bridge prompt:\n{suggested_prompt}\n")
        return (suggested_prompt, report)


# ── Inspector Node ────────────────────────────────────────────────────────────

class GapFillInspector:
    """
    LTX-GapFill Inspector
    ----------------------
    Passthrough debug node. Sits between your image/prompt sources and the
    main GeminiFillPrompt node. Prints a full payload report to the ComfyUI
    console — model, endpoint, frame sizes, prompt text, estimated payload
    size — then passes all inputs through unchanged.

    Insert it in-line without breaking any connections.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    GEMINI_MODELS,
                    {
                        "default": "gemini-2.5-flash",
                        "tooltip": "Gemini model — shown in the report and passed through.",
                    },
                ),
                "custom_model": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Override model dropdown. Leave blank to use dropdown.",
                    },
                ),
                "prompt_style": (
                    PROMPT_STYLES,
                    {
                        "default": "LTX Desktop system prompt",
                        "tooltip": "Shows the full system prompt for this style in the report.",
                    },
                ),
                "gap_duration": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.5,
                        "max": 120.0,
                        "step": 0.5,
                    },
                ),
                "resize_before_send": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "max_size_px": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 64},
                ),
                "print_to_console": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also print the report to the ComfyUI console. Off by default — connect the report output to a Show Text node instead.",
                    },
                ),
            },
            "optional": {
                "clip_before_last_frame":  ("IMAGE",),
                "clip_after_first_frame":  ("IMAGE",),
                "prompt_before":           ("STRING", {"multiline": True, "default": ""}),
                "prompt_after":            ("STRING", {"multiline": True, "default": ""}),
                "custom_system_prompt":    ("STRING", {"multiline": True, "default": ""}),
            },
        }

    # Pass everything through so it can sit in-line
    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("report", "active_model", "prompt_style", "prompt_before", "prompt_after", "clip_before_last_frame", "clip_after_first_frame")
    FUNCTION      = "inspect"
    CATEGORY      = "LTX/Gap Fill"
    OUTPUT_NODE   = True   # always executes even if outputs aren't consumed

    def inspect(
        self,
        model: str,
        custom_model: str,
        prompt_style: str,
        gap_duration: float,
        resize_before_send: bool,
        max_size_px: int,
        print_to_console: bool,
        clip_before_last_frame=None,
        clip_after_first_frame=None,
        prompt_before: str = "",
        prompt_after: str = "",
        custom_system_prompt: str = "",
    ):
        active_model = custom_model.strip() if custom_model.strip() else model
        endpoint     = f"{GEMINI_BASE_URL}/{active_model}:generateContent"

        W = 64  # box inner width

        def _frame_info(tensor, pick_last: bool) -> str:
            if tensor is None:
                return "None"
            idx = -1 if pick_last else 0
            pil = _tensor_to_pil(tensor, idx)
            orig_w, orig_h = pil.size
            if resize_before_send:
                pil = _resize_longest(pil, max_size_px)
            send_w, send_h = pil.size
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="JPEG", quality=85)
            kb = len(buf.getvalue()) / 1024
            batch = tensor.shape[0]
            batch_note = f" (batch={batch}, using {'last' if pick_last else 'first'})" if batch > 1 else ""
            return f"{orig_w}x{orig_h}{batch_note} → sends {send_w}x{send_h} JPEG ~{kb:.0f} KB"

        def _prompt_preview(text: str, max_len: int = W - 4) -> str:
            t = text.strip()
            if not t:
                return "(none)"
            return (t[:max_len - 3] + "...") if len(t) > max_len else t

        def _est_payload_kb() -> float:
            total = 0.0
            for tensor, pick_last in [(clip_before_last_frame, True), (clip_after_first_frame, False)]:
                if tensor is None:
                    continue
                pil = _tensor_to_pil(tensor, -1 if pick_last else 0)
                if resize_before_send:
                    pil = _resize_longest(pil, max_size_px)
                buf = io.BytesIO()
                pil.convert("RGB").save(buf, format="JPEG", quality=85)
                # base64 expands by ~1.33x
                total += len(buf.getvalue()) * 1.33 / 1024
            # rough text overhead
            total += (len(prompt_before) + len(prompt_after) + 800) / 1024
            return total

        # Build the actual system prompt that will be sent — mirrors main node logic
        using_custom = bool(custom_system_prompt.strip())
        if using_custom:
            system_text  = custom_system_prompt.strip()
            max_tokens   = 1024
            style_label  = "⚠ CUSTOM SYSTEM PROMPT (prompt_style ignored)"
        else:
            guidelines   = _STYLE_GUIDELINES.get(prompt_style, _STYLE_GUIDELINES["LTX Desktop system prompt"])
            max_tokens   = 256 if prompt_style == "LTX Desktop system prompt" else 1024
            style_label  = prompt_style
            system_text  = (
                "You are a video production assistant. The user is editing a video timeline and has a gap "
                f"of {gap_duration:.1f} seconds between two shots. Your job is to suggest a detailed prompt "
                "for generating a video clip to fill this gap, so that it flows naturally between the "
                "preceding and following shots.\n\n"
                "Guidelines:\n"
                + guidelines
            )

        before_info = _frame_info(clip_before_last_frame, pick_last=True)
        after_info  = _frame_info(clip_after_first_frame, pick_last=False)
        est_kb      = _est_payload_kb()
        est_str     = f"~{est_kb:.0f} KB" if est_kb < 1024 else f"~{est_kb/1024:.1f} MB"

        div  = "╠" + "═" * W + "╣"

        def row(label: str, value: str) -> str:
            content = f"{label:<16} {value}"
            if len(content) > W - 2:
                content = content[:W - 5] + "..."
            return f"║  {content:<{W - 2}}║"

        def section(title: str) -> str:
            padding = W - len(title) - 4
            return f"║  [{title}]{' ' * max(padding, 0)}║"

        def sys_prompt_lines(text: str) -> list[str]:
            """Wrap system prompt text into box-width lines."""
            out = []
            for raw_line in text.splitlines():
                raw_line = raw_line or ""
                # wrap at W-4 chars
                while len(raw_line) > W - 4:
                    out.append(f"║  {raw_line[:W - 4]:<{W - 2}}║")
                    raw_line = "    " + raw_line[W - 4:]  # indent continuation
                out.append(f"║  {raw_line:<{W - 2}}║")
            return out

        lines = [
            "",
            "╔" + "═" * W + "╗",
            f"║{'  LTX-GapFill Inspector — Payload Report':^{W}}║",
            div,
            row("Model",      active_model),
            row("Endpoint",   endpoint),
            row("Style",      f"{style_label}  (max_tokens={max_tokens})"),
            row("Gap",        f"{gap_duration:.1f} seconds"),
            row("Resize",     f"{'Yes' if resize_before_send else 'No'} (max {max_size_px}px)"),
            row("Est. size",  est_str),
            div,
            section("SYSTEM PROMPT BEING SENT TO GEMINI"),
        ]
        lines += sys_prompt_lines(system_text)
        lines += [
            div,
            section("SHOT BEFORE (clip_before_last_frame)"),
            row("Frame",    before_info),
            row("Prompt",   _prompt_preview(prompt_before)),
            div,
            section("SHOT AFTER  (clip_after_first_frame)"),
            row("Frame",    after_info),
            row("Prompt",   _prompt_preview(prompt_after)),
            "╚" + "═" * W + "╝",
            "",
        ]

        report = "\n".join(lines)

        if print_to_console:
            print(report)

        # Pass-through — downstream GeminiFillPrompt accepts None IMAGE fine.
        return (
            report,
            active_model,
            prompt_style,
            prompt_before,
            prompt_after,
            clip_before_last_frame,
            clip_after_first_frame,
        )


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "GeminiFillPrompt": GeminiFillPrompt,
    "GapFillInspector": GapFillInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiFillPrompt": "LTX-GapFill Prompt (LTX Video Bridge)",
    "GapFillInspector": "LTX-GapFill Inspector",
}
