# ComfyUI-LTX-GapFill

> **This node is designed for Safe For Work (SFW) workflows only.**
> It uses Google Gemini 2.5 Flash, which enforces Google's content policies.
> Do not use this node with NSFW imagery or prompts — requests containing
> inappropriate content will be rejected by the Gemini API.

A ComfyUI custom node that brings the **"Fill with Video"** AI prompt suggestion feature from
[LTX Desktop](https://github.com/Lightricks/LTX-Desktop) into ComfyUI workflows.

> **All credit for the original feature, system prompt design, pipeline architecture, and AI
> integration goes to the [Lightricks](https://github.com/Lightricks) team.** This node is a
> faithful port of their work — the system prompt, Gemini payload structure, and frame analysis
> approach are reproduced verbatim from the LTX Desktop source code. Please support their work:
>
> - LTX Desktop: https://github.com/Lightricks/LTX-Desktop
> - LTX-Video models: https://github.com/Lightricks/LTX-Video
> - Lightricks: https://www.lightricks.com

---

## What It Does

This node uses **Google Gemini 2.5 Flash** to analyze frames and prompts from two neighboring
video clips, then returns a suggested text prompt describing what should go between them.
That prompt is wired into any LTX sampler node to generate the bridging video.

### Use Cases

**Gap filling**
The primary use case — you have two clips on a timeline with empty space between them.
Provide the last frame of clip A and the first frame of clip B, and Gemini suggests a
prompt for a new clip that flows naturally between both shots.

**Transitions**
Feed any two frames from different scenes, shots, or styles to get a prompt for a
transition clip that matches the visual tone of both sides and bridges them smoothly.

**First frame / last frame situations**
Only have one side? The node works one-sided too:
- Supply only `clip_before_last_frame` to generate something that naturally leads *out of* a clip
- Supply only `clip_after_first_frame` to generate something that naturally leads *into* a clip
- Useful for generating an opening shot, a closing shot, or extending a clip at either end

---

## Install

1. Copy this folder into your `ComfyUI/custom_nodes/` directory
2. Install dependencies:
   ```
   pip install requests Pillow numpy
   ```
   (all three are almost certainly already present in a standard ComfyUI environment)
3. Restart ComfyUI

---

## Node: LTX-GapFill Prompt (LTX Video Bridge)

Found under **LTX → Gap Fill** in the node browser.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `seed` | INT | Yes | ComfyUI cache-buster only — **not sent to Gemini**. Change to force re-run when all other inputs are unchanged. |
| `gemini_api_key` | STRING | Yes | Free key from https://aistudio.google.com/app/apikey |
| `model` | DROPDOWN | Yes | Gemini model to use (all free tier with limitations) |
| `custom_model` | STRING | No | Override model dropdown with any Gemini model ID — leave blank to use dropdown |
| `prompt_style` | DROPDOWN | Yes | Controls length and focus of the generated prompt (see Prompt Styles below) |
| `gap_duration` | FLOAT | Yes | Duration of the gap or desired clip length in seconds |
| `resize_before_send` | BOOLEAN | Yes | Resize images before sending (default: True, recommended) |
| `max_size_px` | INT | Yes | Longest edge in pixels when resizing (default: 512) |
| `clip_before_last_frame` | IMAGE | Optional | Connect the clip **before** the gap — last frame auto-selected |
| `clip_after_first_frame` | IMAGE | Optional | Connect the clip **after** the gap — first frame auto-selected |
| `prompt_before` | STRING | Optional | Prompt or description of the clip before the gap |
| `prompt_after` | STRING | Optional | Prompt or description of the clip after the gap |
| `custom_system_prompt` | STRING | Optional | **WARNING:** Overrides `prompt_style` entirely. A poorly written prompt wastes tokens and counts against your daily quota. Leave blank to use the dropdown. |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `suggested_prompt` | STRING | Gemini-generated bridge prompt — wire into CLIP Text Encode or positive prompt |
| `report` | STRING | Full run report showing model, system prompt sent, frames, and Gemini response — wire into Show Text |

### Supplying frames

- **Load Image node** → wire directly into `clip_before_last_frame` or `clip_after_first_frame`
- **VHS LoadVideo** → wire the full IMAGE batch — the node auto-selects:
  - `clip_before_last_frame`: picks the **last** frame from the batch
  - `clip_after_first_frame`: picks the **first** frame from the batch

### Gap fill / transition (both clips)

```
[Load Image A]  ──clip_before_last_frame──►
[Load Image B]  ──clip_after_first_frame──► [LTX-GapFill Prompt] ──suggested_prompt──► [CLIP Text Encode] ──► LTX sampler
                  gap_duration ──►                               ──report──────────────► [Show Text]
                  gemini_api_key ──►
```

### One-sided (opening shot, closing shot, or clip extension)

```
[Load Image A]  ──clip_before_last_frame──► [LTX-GapFill Prompt] ──suggested_prompt──► [CLIP Text Encode] ──► LTX sampler
                  gap_duration ──►
                  gemini_api_key ──►
```

Leave `clip_after_first_frame` or `clip_before_last_frame` disconnected as needed.
Gemini handles the one-sided case and generates something that naturally leads into
or out of the single available shot.

### Re-running without changing inputs

ComfyUI caches node outputs and skips re-running nodes whose inputs haven't changed.
To force a fresh Gemini call, change the `seed` value. Pair it with a **Randomize**
seed node to get a new suggestion on every run automatically.

---

## Node: LTX-GapFill Inspector

Found under **LTX → Gap Fill** in the node browser.

A passthrough debug node for prompt engineers. Sits **in-line** before the main node,
builds and outputs a full payload report showing exactly what will be sent to Gemini —
before anything is sent. All inputs pass through unchanged.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | DROPDOWN | Yes | Gemini model — shown in report and passed through |
| `custom_model` | STRING | Yes | Override model dropdown — leave blank to use dropdown |
| `prompt_style` | DROPDOWN | Yes | Style to preview in the system prompt section of the report |
| `gap_duration` | FLOAT | Yes | Gap duration in seconds |
| `resize_before_send` | BOOLEAN | Yes | Whether images will be resized before sending |
| `max_size_px` | INT | Yes | Longest edge when resizing |
| `print_to_console` | BOOLEAN | Yes | Also print report to ComfyUI console (default: False — use Show Text instead) |
| `clip_before_last_frame` | IMAGE | Optional | Clip before the gap |
| `clip_after_first_frame` | IMAGE | Optional | Clip after the gap |
| `prompt_before` | STRING | Optional | Prompt for clip before gap |
| `prompt_after` | STRING | Optional | Prompt for clip after gap |
| `custom_system_prompt` | STRING | Optional | Custom system prompt override — shown in report if active |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `report` | STRING | Full payload report — wire into Show Text to view inside ComfyUI |
| `active_model` | STRING | Resolved model name — wire into main node's `custom_model` |
| `prompt_style` | STRING | Selected style — wire into main node's `custom_model` is not needed; pass through for reference |
| `prompt_before` | STRING | Passthrough |
| `prompt_after` | STRING | Passthrough |
| `clip_before_last_frame` | IMAGE | Passthrough |
| `clip_after_first_frame` | IMAGE | Passthrough |

### What the report shows

```
╔════════════════════════════════════════════════════════════════╗
║            LTX-GapFill Inspector — Payload Report              ║
╠════════════════════════════════════════════════════════════════╣
║  Model            gemini-2.5-flash                            ║
║  Endpoint         https://generativelanguage.googleapis.com/  ║
║  Style            detailed  (max_tokens=1024)                 ║
║  Gap              5.0 seconds                                 ║
║  Resize           Yes (max 512px)                             ║
║  Est. size        ~108 KB                                     ║
╠════════════════════════════════════════════════════════════════╣
║  [SYSTEM PROMPT BEING SENT TO GEMINI]                         ║
║  You are a video production assistant...                      ║
╠════════════════════════════════════════════════════════════════╣
║  [SHOT BEFORE (clip_before_last_frame)]                       ║
║  Frame            1920x1080 → sends 512x288 JPEG ~42 KB      ║
║  Prompt           A wide shot of a forest at dawn...         ║
╠════════════════════════════════════════════════════════════════╣
║  [SHOT AFTER  (clip_after_first_frame)]                       ║
║  Frame            1920x1080 → sends 512x288 JPEG ~38 KB      ║
║  Prompt           Close-up of sunlight through leaves...     ║
╚════════════════════════════════════════════════════════════════╝
```

### In-line workflow

```
[Load Image A] ──clip_before_last_frame──►
[Load Image B] ──clip_after_first_frame──► [LTX-GapFill Inspector] ──report──────────────────────────────► [Show Text]
                                               ──clip_before_last_frame──►
                                               ──clip_after_first_frame──► [LTX-GapFill Prompt] ──suggested_prompt──► [CLIP Text Encode]
                                               ──prompt_before──────────►                       ──report──────────────► [Show Text]
                                               ──prompt_after───────────►
                                               ──active_model──► (wire to custom_model on main node)
```

---

## Prompt Styles

| Style | Length | Focus |
|-------|--------|-------|
| `LTX Desktop system prompt` | 2-4 sentences | Verbatim LTX Desktop system prompt — matches the app exactly |
| `detailed` | 6-10 sentences | Full description: camera motion, lighting, color palette, atmosphere, depth of field |
| `cinematic` | 5-8 sentences | Cinematographer language: shot type, lens feel, camera movement, lighting ratio, color grade |
| `narrative` | 5-8 sentences | Story-driven: character action, emotional arc, sensory detail, scene purpose |

**Token budget per style:**
- `LTX Desktop system prompt` → 256 max tokens
- all others → 1024 max tokens

---

## Available Models

All models listed below are available on the free tier with usage limitations.

| Model | Notes |
|-------|-------|
| `gemini-2.5-flash` | Default — stable, recommended |
| `gemini-2.5-flash-lite` | Lighter, faster, lower cost |
| `gemini-3-flash-preview` | Newer generation preview |
| `gemini-3.1-flash-lite-preview` | Newest, lightest preview |

Use `custom_model` to enter any model ID not in the dropdown — useful when Google releases new models without a node update.

---

## Why resize?

Gemini has a ~20 MB inline payload limit. A raw 4K PNG encodes to 15–25 MB as base64 —
two of them exceed the limit. At 512 px Gemini has all the scene-level information it
needs (composition, lighting, mood, color palette). Disable `resize_before_send` only if
you specifically need Gemini to read fine text or small details within a frame.

---

## System prompt

Reproduced verbatim from LTX Desktop (`backend/handlers/suggest_gap_prompt_handler.py`),
authored by the Lightricks team:

```
You are a video production assistant. The user is editing a video timeline and
has a gap of {gap_duration} seconds between two shots. Your job is to suggest a
detailed prompt for generating a video clip to fill this gap, so that it flows
naturally between the preceding and following shots.

Guidelines:
- Describe the scene, action, camera movement, lighting, and mood
- Match the visual style and tone of the surrounding shots
- Create a smooth narrative or visual transition between the two shots
- Keep the prompt concise (2-4 sentences max)
- Write only the prompt text, no explanations or labels
- If only one neighboring shot is available, suggest something that naturally
  leads into or out of it
```

---

## Gemini API Key

Get a **free** Gemini API key — no billing or credit card required:

**https://aistudio.google.com/app/apikey**

The free tier is sufficient for this node. It makes one API call per generation,
with no batching. Paste the key directly into the `gemini_api_key` input on the node.

---

## Credits

Original feature design, system prompt, and Gemini integration:
**Lightricks** — https://github.com/Lightricks

This node is a port of their "Fill with Video" feature from LTX Desktop and would
not exist without their research and engineering work on the LTX-Video model family.
