"""Unified LLM client.

Wraps DeepSeek, OpenAI, and Qwen (via OpenRouter, with Alibaba pinned
as the first-party provider) behind a single interface. Swapping
backbones is a CLI ``--provider`` / ``--model`` change.
"""
from __future__ import annotations
import os
import time
import requests


PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "default_model": "deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-5.4",
        "env_key": "OPENAI_API_KEY",
    },
    "qwen": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "qwen/qwen3.5-397b-a17b",
        "env_key": "OPENROUTER_API_KEY",
    },
}

# Pro-tier OpenAI reasoning models (gpt-5*-pro, o1-pro, ...) are only
# served at /v1/responses; chat/completions rejects them.
_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _openai_is_reasoning_model(model: str) -> bool:
    """True for o1/o3/o4/gpt-5* (excluding ``-chat-latest`` chat tunes)."""
    m = model.lower()
    if "chat-latest" in m:
        return False
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    if m.startswith("gpt-5"):
        return True
    return False


def _openai_needs_responses_api(model: str) -> bool:
    # /v1/responses returns reasoning summary text; chat/completions only
    # exposes a token count for reasoning models, which is unhelpful.
    return _openai_is_reasoning_model(model)


_openai_is_reasoning_chat_model = _openai_is_reasoning_model  # back-compat alias


def _deepseek_is_v4_dual_mode(model: str) -> bool:
    # V4-Pro / V4-Flash default to non-thinking; ``thinking={"type":"enabled"}``
    # + ``reasoning_effort`` flips them to chain-of-thought. ``deepseek-reasoner``
    # auto-enables thinking server-side and doesn't need these fields.
    m = model.lower()
    return m in ("deepseek-v4-pro", "deepseek-v4-flash")


REASONING_EFFORTS = ("minimal", "low", "medium", "high")


class LLMClient:
    """Unified DeepSeek / OpenAI / Qwen client.

    Reasoning-model quirks:
    - OpenAI reasoning models ignore ``temperature``/``top_p``.
    - OpenAI pro-tier reasoning models go through /v1/responses in
      background mode; sync would hit Cloudflare's 100s gateway timeout.

    ``effort``       — ``reasoning.effort`` value for thinking-mode backbones.
    ``timeout``      — overall wall-clock budget per background call (s).
    ``poll_interval``— background poll cadence (s).
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "deepseek",
        model: str | None = None,
        timeout: int = 3600,
        poll_interval: float = 10.0,
        effort: str = "medium",
    ):
        if effort not in REASONING_EFFORTS:
            raise ValueError(
                f"effort must be one of {REASONING_EFFORTS}, got {effort!r}"
            )
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]
        self._use_responses_api = (
            provider == "openai" and _openai_needs_responses_api(self.model)
        )
        self.base_url = _OPENAI_RESPONSES_URL if self._use_responses_api else cfg["base_url"]
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.effort = effort
        self.last_usage: dict = {}
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ── public API (unchanged signatures) ────────────────────────────────────

    def call(self, prompt: str | list[dict], temperature: float = 0.0, max_tokens: int = 65536) -> str:
        content, _ = self._call_impl(prompt, temperature, max_tokens)
        return content

    def call_with_reasoning(self, prompt: str | list[dict], temperature: float = 0.0, max_tokens: int = 65536) -> tuple[str, str]:
        return self._call_impl(prompt, temperature, max_tokens)

    # ── internals ────────────────────────────────────────────────────────────

    def _call_impl(self, prompt: str | list[dict], temperature: float = 0.0, max_tokens: int = 65536) -> tuple[str, str]:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

        if self._use_responses_api:
            # /v1/responses in background mode — sync would hit the 100s
            # gateway cap; ``summary=detailed`` keeps the full thinking trace.
            payload = {
                "model": self.model,
                "input": messages,
                "max_output_tokens": max_tokens,
                "reasoning": {"effort": self.effort, "summary": "detailed"},
                "background": True,
            }
            return self._call_background(payload)

        payload = {
            "model": self.model,
            "messages": messages,
        }
        # OpenAI reasoning models use max_completion_tokens and need an
        # explicit reasoning_effort, otherwise gpt-5* silently runs at
        # zero reasoning tokens. They also ignore temperature/top_p.
        if self.provider == "openai":
            payload["max_completion_tokens"] = max_tokens
            if _openai_is_reasoning_chat_model(self.model):
                payload["reasoning_effort"] = self.effort
            else:
                payload["top_p"] = 1.0
        elif self.provider == "qwen":
            # Both `reasoning_effort` (legacy) and `reasoning.effort` (new)
            # are sent; OpenRouter forwards whichever each backend accepts.
            # Pin first-party Alibaba (override via OPENROUTER_PROVIDER_ONLY).
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
            payload["top_p"] = 1.0
            payload["reasoning_effort"] = self.effort
            payload["reasoning"] = {"effort": self.effort}
            env_only = os.environ.get("OPENROUTER_PROVIDER_ONLY", "").strip()
            if env_only:
                provider_only = [s.strip() for s in env_only.split(",") if s.strip()]
            else:
                provider_only = ["Alibaba", "Alibaba Cloud", "DashScope"]
            payload["provider"] = {
                "only": provider_only,
                "allow_fallbacks": False,
            }
        elif _deepseek_is_v4_dual_mode(self.model):
            # V4-Pro / V4-Flash: opt in to thinking mode; sampling fields
            # are silently ignored once enabled.
            payload["max_tokens"] = max_tokens
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = self.effort
        else:
            payload["temperature"] = temperature
            payload["top_p"] = 1.0
            payload["max_tokens"] = max_tokens

        return self._call_sync(payload)

    def _call_sync(self, payload: dict) -> tuple[str, str]:
        """Synchronous chat/completions path with bounded backoff retries."""
        max_attempts = 5
        base_delay = 2.0

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    self.base_url, headers=self.headers, json=payload, timeout=self.timeout,
                )

                if response.status_code in [429, 500, 502, 503, 504]:
                    delay = base_delay * (2 ** attempt)
                    print(f"[API WARN] HTTP {response.status_code}. Retrying in {delay}s... ({attempt+1}/{max_attempts})")
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError:
                    data = None
                if not isinstance(data, dict):
                    # DeepSeek occasionally returns 200 with a null body;
                    # treat as transient.
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"[API WARN] Empty/null response body. Retrying in {delay}s... ({attempt+1}/{max_attempts})")
                        time.sleep(delay)
                        continue
                    raise RuntimeError("API returned empty/null body after retries")
                self.last_usage = data.get("usage", {}) or {}

                choices = data.get("choices") or []
                if not choices:
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"[API WARN] No choices in response. Retrying in {delay}s... ({attempt+1}/{max_attempts})")
                        time.sleep(delay)
                        continue
                    raise RuntimeError("Unexpected API response format: no choices")

                message = choices[0].get("message") or {}
                content = message.get("content", "") or ""
                reasoning = (
                    message.get("reasoning_content", "")
                    or message.get("reasoning", "")
                    or ""
                )

                if content:
                    return content, reasoning
                elif reasoning:
                    return reasoning, ""
                else:
                    raise RuntimeError("No content in API response")

            except requests.exceptions.RequestException as e:
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[API WARN] Network error: {e}. Retrying in {delay}s... ({attempt+1}/{max_attempts})")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"API call failed: {e}")
            except (KeyError, IndexError, AttributeError, TypeError) as e:
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[API WARN] Response format error: {e}. Retrying in {delay}s... ({attempt+1}/{max_attempts})")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Unexpected API response format: {e}")

        raise RuntimeError(f"API call failed after {max_attempts} attempts")

    def _call_background(self, payload: dict) -> tuple[str, str]:
        """Submit to /v1/responses with ``background=true`` and poll.

        Submit-side errors retry with exponential backoff. Poll-side errors
        retry in place; the in-flight job is left running.
        """
        # 1. Submit the job.
        submit_max_attempts = 5
        submit_base_delay = 2.0
        response_id: str | None = None
        status: str | None = None

        for attempt in range(submit_max_attempts):
            try:
                resp = requests.post(
                    self.base_url, headers=self.headers, json=payload, timeout=60,
                )
                if resp.status_code in [429, 500, 502, 503, 504]:
                    delay = submit_base_delay * (2 ** attempt)
                    print(f"[API WARN] Submit HTTP {resp.status_code}. Retrying in {delay}s... ({attempt+1}/{submit_max_attempts})")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                response_id = data.get("id")
                status = data.get("status")
                if not response_id:
                    raise RuntimeError(f"Background submit returned no id: {data}")
                break
            except requests.exceptions.RequestException as e:
                if attempt < submit_max_attempts - 1:
                    delay = submit_base_delay * (2 ** attempt)
                    print(f"[API WARN] Submit network error: {e}. Retrying in {delay}s... ({attempt+1}/{submit_max_attempts})")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Background submit failed: {e}")

        if response_id is None:
            raise RuntimeError("Background submit failed after all retries")

        print(f"[API BG] Submitted {response_id} (status={status}), polling every {self.poll_interval}s")

        # 2. Poll until a terminal status.
        poll_url = f"{self.base_url}/{response_id}"
        start_time = time.monotonic()
        consecutive_poll_errors = 0
        max_consecutive_poll_errors = 10
        data: dict = {}
        last_logged_status: str | None = None

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > self.timeout:
                # Cancel so we don't orphan a paid job.
                try:
                    requests.post(
                        f"{poll_url}/cancel", headers=self.headers, timeout=30,
                    )
                except Exception:
                    pass
                raise RuntimeError(
                    f"Background job {response_id} exceeded timeout of {self.timeout}s"
                )

            time.sleep(self.poll_interval)

            try:
                resp = requests.get(poll_url, headers=self.headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                consecutive_poll_errors = 0
            except requests.exceptions.RequestException as e:
                consecutive_poll_errors += 1
                print(f"[API WARN] Poll error ({consecutive_poll_errors}/{max_consecutive_poll_errors}): {e}")
                if consecutive_poll_errors >= max_consecutive_poll_errors:
                    raise RuntimeError(f"Too many consecutive poll errors on {response_id}: {e}")
                continue

            status = data.get("status")
            if status != last_logged_status:
                print(f"[API BG] {response_id} status={status} (elapsed {elapsed:.0f}s)")
                last_logged_status = status

            if status == "completed":
                break
            if status in ("failed", "cancelled", "expired"):
                err = data.get("error") or data.get("incomplete_details") or data
                raise RuntimeError(f"Background job {response_id} ended with status={status}: {err}")
            if status == "incomplete":
                details = data.get("incomplete_details") or {}
                reason = details.get("reason", "unknown")
                # `incomplete` covers max_output_tokens; return what we got.
                print(f"[API WARN] Background job {response_id} incomplete: {reason}")
                break
            # queued / in_progress: keep polling.

        # 3. Parse terminal response.
        self.last_usage = data.get("usage", {})
        content, reasoning = self._parse_responses_output(data)
        if content:
            return content, reasoning
        if reasoning:
            return reasoning, ""
        raise RuntimeError(f"No content in background response {response_id}: {data}")

    def get_normalized_usage(self) -> dict:
        """Cross-provider token usage from the last call."""
        u = self.last_usage
        return {
            "input_tokens": u.get("input_tokens") or u.get("prompt_tokens", 0),
            "output_tokens": u.get("output_tokens") or u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
            "reasoning_tokens": (
                (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                or (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            ),
        }

    @staticmethod
    def _parse_responses_output(data: dict) -> tuple[str, str]:
        """Extract (content, reasoning_summary) from a /v1/responses payload."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in data.get("output", []) or []:
            itype = item.get("type")
            if itype == "message":
                for c in item.get("content", []) or []:
                    if c.get("type") == "output_text":
                        content_parts.append(c.get("text", "") or "")
            elif itype == "reasoning":
                for s in item.get("summary", []) or []:
                    if isinstance(s, dict) and s.get("type") == "summary_text":
                        reasoning_parts.append(s.get("text", "") or "")
        return "".join(content_parts), "\n\n".join(reasoning_parts)


# Back-compat alias.
DeepSeekClient = LLMClient
