import os
import json
import time
import io
import requests  # noqa: F401
import numpy as np
from PIL import Image

_p = os.path.dirname(os.path.realpath(__file__))

MODEL_FALLBACKS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def _get_config():
    try:
        path = os.path.join(_p, "config.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(cfg: dict):
    try:
        path = os.path.join(_p, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


class ComfyUIGeminiPromptBuilderText:
    """
    Gemini text model + Google Search grounding.
    Output: prompt text only (no JSON).
    Editable System Instruction in UI.
    """

    def __init__(self, apikey=None):
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINIAPIKEY")
        placeholders = {"tokenhere", "placetokenhere", "yourapikey", "apikeyhere", "enteryourkey", "", None}
        if isinstance(env_key, str) and env_key.strip().lower() not in placeholders:
            self.apikey = env_key.strip()
        else:
            self.apikey = apikey.strip() if isinstance(apikey, str) else None

        if not self.apikey:
            cfg = _get_config()
            self.apikey = cfg.get("GEMINI_API_KEY") or cfg.get("GEMINIAPIKEY")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    @classmethod
    def INPUT_TYPES(cls):
        # 默认指令：整合了最新的“高审美 + 反AI”逻辑
        default_instruction = (
            "你是“出图提示词编排器”，专精为 gemini-3-pro-image-preview 生成用于发布到快手的写实美女摄影作品提示词。你的唯一产出：根据输入生成一条可直接用于图像生成的最终提示词（单段纯文本，禁止换行）。\n\n"
            "【任务核心】：利用 Google Search 检索最新的“快手/小红书爆款生活感”摄影风格。我们要的是“高审美素人感”：利用“真实的摄影瑕疵”来对抗 AI 的工业完美感。\n\n"
            "【审美红线】：\n"
            "1. **拒绝土味**：严禁描述脏、乱、旧的衣物或场景。\n"
            "2. **追求质感**：场景应为“有审美的极简/温馨/ins风”。穿搭应为“有质感的莫代尔、羊绒、纯棉”。\n"
            "3. **光影进阶**：优先使用“混合光”。例如：暖色台灯 + 窗户进来的冷色残余光。\n\n"
            "【约束模板】：\n"
            "[IDENTITY_STRICT_ANCHOR]: (The exact specific woman from reference images:1.6), (absolute facial consistency:1.5), (170cm height, photorealistic slender figure:1.5).\n"
            "[CREATIVE_DIRECTOR_MANDATE]: Act as an aesthetic lifestyle blogger on Kuaishou. Priorities: 1) SAME FACE identity, 2) high-end cozy vibe, 3) raw snapshot authenticity.\n"
            "[CORE_STYLE]: (Candid smartphone photo:1.25), (unfiltered life moment:1.2), (slight motion blur on hands:1.1), (natural sensor noise:1.1).\n"
            "[WARDROBE_RULES]: <写一套燕麦色或奶油白系的、显质感的松弛穿搭>.\n"
            "[SCENE_AUTOPILOT]: <写一个审美在线的写实生活场景（如：木质书桌、暖色台灯、简约窗台）>.\n"
            "[ACTION_AUTOPILOT]: <写一个沉浸式的自然动作>.\n"
            "[CAMERA_DIVERSITY]: <写一个随手拍视角：略微歪斜、主体偏离中心>.\n"
            "[LIGHTING_PHYSICS]: <写一套有层次的混合光>.\n"
            "[NEGATIVE_CONSTRAINTS]: (CGI, 3D render, plastic skin:1.8), (cluttered dirty background:1.6), (perfect symmetry, professional studio lighting:1.6)."
        )

        return {
            "required": {
                "goal": ("STRING", {"default": "运动松弛感，书桌前。", "multiline": True}),
                "system_instruction": ("STRING", {"default": default_instruction, "multiline": True}),
                "model": (MODEL_FALLBACKS, {"default": MODEL_FALLBACKS[0]}),
                "apikey": ("STRING", {"default": ""}),
                "enable_search": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 64}),
                "max_attempts": ("INT", {"default": 3, "min": 1, "max": len(MODEL_FALLBACKS), "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "operation_log")
    FUNCTION = "build_prompt"
    CATEGORY = "Gemini / Prompt"

    def _build_config_best_effort(self, types, enable_search, temperature, seed, top_p, top_k, max_output_tokens):
        tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search else None
        base = {
            "tools": tools,
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "seed": int(seed)
        }
        try:
            return types.GenerateContentConfig(**base)
        except Exception:
            return base

    def _tensor_to_pil(self, image_tensor):
        """
        将 ComfyUI 的 IMAGE tensor 转换为 PIL Image。
        ComfyUI IMAGE 格式: [B, H, W, C], 值范围 0-1, float32
        """
        # 取第一张图片 (batch 中的第一个)
        img = image_tensor[0]
        # 转换为 numpy 并缩放到 0-255
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _pil_to_base64(self, pil_image, format="PNG"):
        """
        将 PIL Image 转换为 base64 编码的字节数据。
        """
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()

    def _extract_grounding_log(self, response):
        lines = []
        try:
            cand = response.candidates[0] if hasattr(response, "candidates") and response.candidates else response
            gm = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
            if gm:
                lines.append("Grounding meta <present>")
                queries = getattr(gm, "web_search_queries", None) or getattr(gm, "webSearchQueries", None)
                if queries:
                    for q in queries: lines.append(f"- Search Query: {q}")
        except Exception: pass
        return lines

    def _build_model_attempts(self, model, max_attempts):
        ordered = [model] if model else []
        for candidate in MODEL_FALLBACKS:
            if candidate not in ordered:
                ordered.append(candidate)
        try:
            limit = int(max_attempts)
        except Exception:
            limit = len(ordered)
        limit = max(1, min(limit, len(ordered)))
        return ordered[:limit]

    def build_prompt(self, goal, system_instruction, model, apikey, enable_search, temperature, seed, top_p, top_k, max_output_tokens, max_attempts, image=None):
        if isinstance(apikey, str) and apikey.strip():
            self.apikey = apikey.strip()
            _save_config({"GEMINI_API_KEY": self.apikey})
        if not self.apikey: return "", "ERROR: Missing API key."

        try:
            from google import genai
            from google.genai import types
        except Exception: return "", "ERROR: google-genai not installed."

        config = self._build_config_best_effort(types, enable_search, temperature, seed, top_p, top_k, max_output_tokens)

        client = genai.Client(api_key=self.apikey)
        # 使用用户在 UI 中填写的 system_instruction
        combined = f"{system_instruction.strip()}\n\n用户输入目标：{goal.strip()}"

        # 构建 contents：如果有图片则创建 multimodal 内容
        if image is not None:
            # 将 ComfyUI IMAGE tensor 转换为 PIL Image
            pil_img = self._tensor_to_pil(image)
            # 转换为字节数据
            img_bytes = self._pil_to_base64(pil_img, format="PNG")
            # 创建 multimodal contents: 图片 + 文本
            contents = [
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                combined  # 直接使用字符串，SDK 会自动处理
            ]
        else:
            contents = [combined]

        attempts = self._build_model_attempts(model, max_attempts)
        last_error = None
        for attempt_index, attempt_model in enumerate(attempts, start=1):
            try:
                resp = client.models.generate_content(model=attempt_model, contents=contents, config=config)
                text = (getattr(resp, "text", None) or "").strip()

                # 清理换行，确保输出为一行
                text = " ".join(text.splitlines()).strip()

                op = ["GEMINI PROMPT BUILDER LOG", f"Model: {attempt_model}"]
                if image is not None:
                    op.append("Image input: <provided>")
                if attempt_index > 1:
                    op.append(f"Retry: {attempt_index}/{len(attempts)}")
                op.extend(self._extract_grounding_log(resp))
                return text, "\n".join(op)
            except Exception as e:
                last_error = e
                continue

        last_model = attempts[-1] if attempts else model
        return "", f"ERROR: {str(last_error)} (model={last_model}, attempts={len(attempts)})"


NODE_CLASS_MAPPINGS = {"ComfyUIGeminiPromptBuilderText": ComfyUIGeminiPromptBuilderText}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUIGeminiPromptBuilderText": "Gemini Prompt Builder (Editable Instruction)"}
