# mh_openrouter.py
# Lightweight OpenRouter API client for MHSafeEval.
#
# fix: counselor model은 call_counselor()에 인자로 전달받도록 변경
#      (기존 COUNSELOR_MODEL 하드코딩 → 평가 대상 모델을 외부에서 지정 가능)
#      JUDGE_MODEL / ATTACK_MODEL 은 gpt-4o-mini 고정 유지

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL    = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RETRY   = 3
_RETRY_SLEEP = 5

# ── Fixed models (attacker, judge) ────────────────────────────────────────────
ATTACK_MODEL  = "openai/gpt-4o-mini"   # patient 역할 attack model (고정)
JUDGE_MODEL   = "openai/gpt-4o-mini"   # judge model (고정)

# counselor model 기본값 (args.counselor_model이 없을 때 fallback용)
DEFAULT_COUNSELOR_MODEL = "openai/gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# Raw call
# ─────────────────────────────────────────────────────────────────────────────

def _call(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/MHSafeEval",
        "X-Title":       "MHSafeEval",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
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
# Counselor (target model — configurable, 평가 대상)
# ─────────────────────────────────────────────────────────────────────────────

def call_counselor(
    system_prompt:    str,
    client_utterance: str,
    max_tokens:       int = 300,
    counselor_model:  str = None,   # None → DEFAULT_COUNSELOR_MODEL
) -> str:
    """
    Call the counselor (target) model via OpenRouter.
    counselor_model is the model being evaluated — passed from args.counselor_model.
    """
    model = counselor_model or DEFAULT_COUNSELOR_MODEL
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": client_utterance},
    ]
    return _call(model, messages, max_tokens=max_tokens, temperature=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Judge (fixed to JUDGE_MODEL)
# ─────────────────────────────────────────────────────────────────────────────

def call_judge(
    system_prompt: str,
    client_msg:    str,
    counselor_resp: str,
    max_tokens:    int = 120,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Client: {client_msg}\n\nCounselor: {counselor_resp}"},
    ]
    return _call(JUDGE_MODEL, messages, max_tokens=max_tokens, temperature=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Attack model (fixed to ATTACK_MODEL — patient 역할)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
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
    Manages the PAIR-style iterative refinement loop for the attack model (patient).
    attack model은 ATTACK_MODEL(gpt-4o-mini)로 고정.
    """

    def __init__(
        self,
        model_name:    str,       # 사용은 하되 ATTACK_MODEL로 override됨
        system_prompt: str,
        init_msg:      str,
        max_tokens:    int   = 500,
        temperature:   float = 1.0,
        keep_last_n:   int   = 4,
    ):
        # attack model은 항상 ATTACK_MODEL로 고정
        self.model_name  = ATTACK_MODEL
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.keep_last_n = keep_last_n

        self._system       = system_prompt
        self._history:     list[dict] = []
        self._pending_user = init_msg

    def get_attack(self, feedback: str | None = None) -> dict | None:
        user_msg = feedback if feedback is not None else self._pending_user

        trimmed  = self._history[-(2 * self.keep_last_n):] if self._history else []
        messages = (
            [{"role": "system", "content": self._system}]
            + trimmed
            + [{"role": "user", "content": user_msg}]
        )

        raw = _call(self.model_name, messages, self.max_tokens, self.temperature)
        if not raw:
            return None

        parsed = _extract_json(raw)
        if parsed is None:
            print(f"[AttackModel] JSON parse failed. Raw:\n{raw[:300]}")
            return None

        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant",  "content": raw})

        return parsed