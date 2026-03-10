# ComfyUI-LTX-GapFill

> **This node is designed for Safe For Work (SFW) workflows only.**
> It uses Google Gemini 2.0 Flash, which enforces Google's content policies.
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

This node uses **Google Gemini 2.0 Flash** to analyze frames and prompts from two neighboring
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
- Supply only `last_frame` to generate something that naturally leads *out of* a clip
- Supply only `first_frame` to generate something that naturally leads *into* a clip
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

## Node: Gemini Fill Prompt (LTX Bridge)

Found under **LTX → Gap Fill** in the node browser.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `gemini_api_key` | STRING | Yes | Free key from https://aistudio.google.com/app/apikey |
| `gap_duration` | FLOAT | Yes | Duration of the gap or desired clip length in seconds |
| `resize_before_send` | BOOLEAN | Yes | Resize images before sending (default: True, recommended) |
| `max_size_px` | INT | Yes | Longest edge in pixels when resizing (default: 512) |
| `last_frame` | IMAGE | Optional | Last frame of the clip BEFORE the gap |
| `first_frame` | IMAGE | Optional | First frame of the clip AFTER the gap |
| `prompt_before` | STRING | Optional | Prompt or description of the clip before the gap |
| `prompt_after` | STRING | Optional | Prompt or description of the clip after the gap |

### Output

| Output | Type | Description |
|--------|------|-------------|
| `suggested_prompt` | STRING | Gemini-generated bridge prompt, ready for CLIP Text Encode |

---

## Usage

### Supplying frames

- **Load Image node** → wire directly into `last_frame` or `first_frame`
- **VHS LoadVideo** → wire the IMAGE batch output directly — the node auto-selects:
  - `last_frame` input: picks the **last** frame from the batch
  - `first_frame` input: picks the **first** frame from the batch

### Gap fill / transition (both clips)

```
[Load Image A]  ──last_frame──►
[Load Image B]  ──first_frame──► [Gemini Fill Prompt] ──► [CLIP Text Encode] ──► LTX sampler
                  gap_duration ──►
                  gemini_api_key ──►
```

### One-sided (opening shot, closing shot, or clip extension)

```
[Load Image A]  ──last_frame──► [Gemini Fill Prompt] ──► [CLIP Text Encode] ──► LTX sampler
                  gap_duration ──►
                  gemini_api_key ──►
```

Leave `first_frame` / `last_frame` disconnected as needed. Gemini's system prompt handles
the one-sided case and generates something that naturally leads into or out of the
single available shot.

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
