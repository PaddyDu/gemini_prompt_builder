import os
import json
import time
import io
import base64
import numpy as np
from PIL import Image

_p = os.path.dirname(os.path.realpath(__file__))


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
        existing = _get_config()
        existing.update(cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


class ComfyUIOpenAIPromptBuilderText:
    """
    OpenAI-compatible text model prompt builder.
    Supports custom base URL for self-hosted / third-party OpenAI-compatible APIs.
    Output: prompt text only (no JSON).
    Editable System Instruction in UI.
    """

    def __init__(self, apikey=None):
        env_key = os.environ.get("OPENAI_API_KEY")
        placeholders = {"tokenhere", "placetokenhere", "yourapikey", "apikeyhere", "enteryourkey", "", None}
        if isinstance(env_key, str) and env_key.strip().lower() not in placeholders:
            self.apikey = env_key.strip()
        else:
            self.apikey = apikey.strip() if isinstance(apikey, str) else None

        if not self.apikey:
            cfg = _get_config()
            self.apikey = cfg.get("OPENAI_API_KEY")

        cfg = _get_config()
        self.base_url = cfg.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    @classmethod
    def INPUT_TYPES(cls):
        default_instruction = (
            "你是【出图提示词编排器】，专精为图像生成模型生成用于发布到快手的写实美女摄影作品提示词。你的唯一产出：根据输入生成一条可直接用于图像生成的最终提示词（单段纯文本，禁止换行）。\n\n"
            "【任务核心】：我们要的是【高审美素人感】：利用【真实的摄影瑕疵】来对抗 AI 的工业完美感。\n\n"
            "【审美红线】：\n"
            "1. **拒绝土味**：严禁描述脏乱旧的衣物或场景。\n"
            "2. **追求质感**：场景应为【有审美的极简/温馨/ins风】。穿搭应为【有质感的莫代尔/羊绒/纯棉】。\n"
            "3. **光影进阶**：优先使用【混合光】。例如：暖色台灯 + 窗户进来的冷色残余光。\n\n"
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
                "model": ("STRING", {"default": "gpt-4o", "multiline": False}),
                "apikey": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 64}),
                "max_attempts": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "operation_log")
    FUNCTION = "build_prompt"
    CATEGORY = "OpenAI / Prompt"

    def _tensor_to_pil(self, image_tensor):
        """
        将 ComfyUI 的 IMAGE tensor 转换为 PIL Image。
        ComfyUI IMAGE 格式: [B, H, W, C], 值范围 0-1, float32
        """
        img = image_tensor[0]
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _pil_to_base64(self, pil_image, fmt="PNG"):
        """将 PIL Image 转换为 base64 字符串（用于 OpenAI vision API）。"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_with_fallback(self, client, resolved_model, messages, max_tokens, temperature, top_p, seed):
        """
        尝试多种参数组合，对不兼容的参数做降级处理。
        返回 (response, used_param_keys) 或抛出最后一个异常。
        """
        param_sets = [
            # level 0: 全参数
            {"max_tokens": int(max_tokens), "temperature": float(temperature), "top_p": float(top_p), "seed": int(seed)},
            # level 1: 去掉 seed（很多第三方不支持）
            {"max_tokens": int(max_tokens), "temperature": float(temperature), "top_p": float(top_p)},
            # level 2: 去掉 seed + top_p
            {"max_tokens": int(max_tokens), "temperature": float(temperature)},
            # level 3: 只保留 max_tokens（最大兼容性）
            {"max_tokens": int(max_tokens)},
        ]
        last_exc: Exception = RuntimeError("No parameter set succeeded.")
        for params in param_sets:
            try:
                resp = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    **params,
                )
                return resp, list(params.keys())
            except Exception as e:
                err_str = str(e)
                # 认证/授权错误直接抛出，不继续降级
                if "401" in err_str or "403" in err_str or "invalid_api_key" in err_str.lower():
                    raise
                # 只对 400（参数不兼容）做降级
                if "400" in err_str:
                    last_exc = e
                    continue
                # 其他错误（超时、网络等）直接抛出
                raise
        raise last_exc

    def build_prompt(self, goal, system_instruction, model, apikey, base_url,
                     temperature, seed, top_p, max_tokens, max_attempts, image=None):
        # 更新 API key
        if isinstance(apikey, str) and apikey.strip():
            self.apikey = apikey.strip()
            _save_config({"OPENAI_API_KEY": self.apikey})

        if not self.apikey:
            return "", "ERROR: Missing API key. Please set OPENAI_API_KEY in environment or enter in the node."

        # 更新 base URL
        resolved_base_url = base_url.strip() if isinstance(base_url, str) and base_url.strip() else "https://api.openai.com/v1"
        if resolved_base_url != self.base_url:
            self.base_url = resolved_base_url
            _save_config({"OPENAI_BASE_URL": self.base_url})

        try:
            from openai import OpenAI
        except ImportError:
            return "", "ERROR: openai package not installed. Run: pip install openai"

        client = OpenAI(api_key=self.apikey, base_url=self.base_url)

        resolved_model = model.strip() if isinstance(model, str) and model.strip() else "gpt-4o"

        # 构建 messages
        messages = [
            {"role": "system", "content": system_instruction.strip()},
        ]

        if image is not None:
            pil_img = self._tensor_to_pil(image)
            b64_img = self._pil_to_base64(pil_img, fmt="PNG")
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": f"用户输入目标：{goal.strip()}"
                }
            ]
        else:
            user_content = f"用户输入目标：{goal.strip()}"

        messages.append({"role": "user", "content": user_content})

        last_error: Exception = RuntimeError("Unknown error.")
        for attempt in range(1, int(max_attempts) + 1):
            try:
                resp, used_params = self._call_with_fallback(
                    client, resolved_model, messages,
                    max_tokens, temperature, top_p, seed
                )

                text = (resp.choices[0].message.content or "").strip()
                text = " ".join(text.splitlines()).strip()

                usage = resp.usage
                usage_info = ""
                if usage:
                    usage_info = f"Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"

                op_lines = [
                    "OPENAI PROMPT BUILDER LOG",
                    f"Model: {resolved_model}",
                    f"Base URL: {self.base_url}",
                    f"Params used: {', '.join(used_params)}",
                ]
                if image is not None:
                    op_lines.append("Image input: <provided>")
                if attempt > 1:
                    op_lines.append(f"Retry: {attempt}/{max_attempts}")
                if usage_info:
                    op_lines.append(usage_info)

                return text, "\n".join(op_lines)

            except Exception as e:
                last_error = e
                err_str = str(e)
                if "401" in err_str or "403" in err_str or "invalid_api_key" in err_str.lower():
                    break
                continue

        return "", f"ERROR after {max_attempts} attempt(s): {str(last_error)} (model={resolved_model}, base_url={self.base_url})"


NODE_CLASS_MAPPINGS = {"ComfyUIOpenAIPromptBuilderText": ComfyUIOpenAIPromptBuilderText}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUIOpenAIPromptBuilderText": "OpenAI Prompt Builder (Custom Base URL)"}
