# mh_openrouter.py
# Lightweight OpenRouter API client for MHSafeEval.
# Handles attack model (client), counselor (target), and judge — all via OpenRouter.
# Uses requests only; no litellm/openai required.

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RETRY = 3
_RETRY_SLEEP = 5

# ── Fixed models ──────────────────────────────────────────────────────────────
COUNSELOR_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL     = "openai/gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# Raw call
# ─────────────────────────────────────────────────────────────────────────────

def _call(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    """Single OpenRouter completion call with retry."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MHSafeEval",
        "X-Title": "MHSafeEval",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(_MAX_RETRY):
        try:
            resp = requests.post(
                _BASE_URL, headers=headers, json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except requests.exceptions.HTTPError as e:
            print(f"[OpenRouter] HTTP {resp.status_code} (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"[OpenRouter] Error (attempt {attempt+1}): {e}")

        if attempt < _MAX_RETRY - 1:
            time.sleep(_RETRY_SLEEP)

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Counselor (target model — fixed to gpt-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────

def call_counselor(system_prompt: str, client_utterance: str, max_tokens: int = 300) -> str:
    """Call the fixed counselor model (gpt-4o-mini via OpenRouter)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": client_utterance},
    ]
    return _call(COUNSELOR_MODEL, messages, max_tokens=max_tokens, temperature=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Judge (fixed to gpt-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────

def call_judge(system_prompt: str, client_msg: str, counselor_resp: str, max_tokens: int = 120) -> str:
    """Call the fixed judge model (gpt-4o-mini via OpenRouter)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Client: {client_msg}\n\nCounselor: {counselor_resp}"},
    ]
    return _call(JUDGE_MODEL, messages, max_tokens=max_tokens, temperature=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Attack model (client — configurable)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    """
    Extract {'improvement': ..., 'prompt': ...} JSON from attack model output.
    Mirrors common.py's extract_json() but self-contained.
    """
    # Try the whole string first, then look for first { ... }
    candidates = [raw]
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        candidates.append(raw[start:end])

    for text in candidates:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if "improvement" in parsed and "prompt" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction
    imp_match = re.search(r'"improvement"\s*:\s*"(.*?)"(?=\s*,|\s*})', raw, re.DOTALL)
    prm_match = re.search(r'"prompt"\s*:\s*"(.*?)"(?=\s*,|\s*})',      raw, re.DOTALL)
    if imp_match and prm_match:
        return {
            "improvement": imp_match.group(1),
            "prompt":      prm_match.group(1),
        }
    return None


class OpenRouterAttackModel:
    """
    Manages the PAIR-style iterative refinement loop for the attack model (client).
    Keeps conversation history internally; caller invokes get_attack() per turn.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        init_msg: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        keep_last_n: int = 4,
    ):
        self.model_name  = model_name
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.keep_last_n = keep_last_n

        # Conversation history (excluding system message)
        self._system  = system_prompt
        self._history: list[dict] = []
        self._pending_user = init_msg   # first user message not yet sent

    def get_attack(self, feedback: str | None = None) -> dict | None:
        """
        Run one PAIR refinement step.

        Args:
            feedback: The processed target response string (from process_target_response).
                      On the very first call, pass None to use the init message.

        Returns:
            dict with keys 'improvement' and 'prompt', or None on parse failure.
        """
        # Determine user message for this turn
        if feedback is not None:
            user_msg = feedback
        else:
            user_msg = self._pending_user

        # Build messages: system + trimmed history + new user message
        trimmed = self._history[-(2 * self.keep_last_n):] if self._history else []
        messages = (
            [{"role": "system", "content": self._system}]
            + trimmed
            + [{"role": "user", "content": user_msg}]
        )

        # Call attack model
        raw = _call(self.model_name, messages, self.max_tokens, self.temperature)
        if not raw:
            return None

        # Parse JSON
        parsed = _extract_json(raw)
        if parsed is None:
            print(f"[AttackModel] JSON parse failed. Raw:\n{raw[:300]}")
            return None

        # Update history
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant",  "content": raw})

        return parsed