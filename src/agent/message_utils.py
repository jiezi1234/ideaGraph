from __future__ import annotations

import json
import re


def strip_json_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_from_content(content: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", (content or "").strip(), flags=re.DOTALL).strip()
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        text = code_block.group(1).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        # 模型有时会先解释再给 JSON，或在 JSON 后补一句话；只取第一个完整 JSON 载荷。
        return text[index : index + end].strip()
    return text
