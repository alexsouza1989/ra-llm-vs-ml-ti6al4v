"""API clients for the three LLMs, keyed from environment variables.

Required environment variables (see .env.example):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

Model identifiers used in the study:
    Claude : claude-sonnet-4-5
    GPT    : gpt-4o-mini
    Gemini : gemini-3-flash-preview

All calls use temperature = 0. max_tokens: 32 (few-shot) / 512 (CoT).
"""

import os

MODEL_CLAUDE = "claude-sonnet-4-5"
MODEL_GPT = "gpt-4o-mini"
MODEL_GEMINI = "gemini-3-flash-preview"


def _require_key(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if not key:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"Copy .env.example to .env and fill in your keys, "
            f"or export {name} in your shell."
        )
    return key


def call_claude(prompt: str, system: str,
                model: str = MODEL_CLAUDE, max_tokens: int = 32) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:  # noqa: BLE001 — keep LOOCV running on failures
        print(f"    [Claude ERROR] {e}")
        return "nan"


def call_gpt(prompt: str, system: str,
             model: str = MODEL_GPT, max_tokens: int = 32) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        print(f"    [GPT ERROR] {e}")
        return "nan"


def call_gemini(prompt: str, system: str,
                model: str = MODEL_GEMINI, max_tokens: int = 32) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=_require_key("GEMINI_API_KEY"))
        m = genai.GenerativeModel(model)
        resp = m.generate_content(
            f"{system}\n\n{prompt}",
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
        return resp.text.strip()
    except Exception as e:  # noqa: BLE001
        print(f"    [Gemini ERROR] {e}")
        return "nan"


CALLERS = {
    "claude": call_claude,
    "gpt": call_gpt,
    "gemini": call_gemini,
}
