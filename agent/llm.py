"""OpenAI-compatible chat client with TOOL CALLING, retries, and friendly errors (Groq, OpenRouter, OpenAI...)."""
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class ChatError(Exception):
    def __init__(self, user_message: str, retryable: bool = False):
        super().__init__(user_message)
        self.user_message = user_message
        self.retryable = retryable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON string exactly as the model produced it


@dataclass
class LLMReply:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(cls, secrets=None) -> Optional["LLMConfig"]:
        """None (not an error) when no key is set: the app then offers demo mode."""
        secrets = secrets or {}
        get = lambda n, d=None: (os.getenv(n) or secrets.get(n) or d)  # noqa: E731
        key = (get("LLM_API_KEY") or "").strip()
        if not key:
            return None
        return cls(api_key=key, base_url=get("LLM_BASE_URL", cls.base_url).strip(),
                   model=get("LLM_MODEL", cls.model).strip())


def classify_error(exc: Exception) -> ChatError:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    detail = str(exc)[:200]
    if "tool_use_failed" in str(exc):     # model produced a malformed tool call: worth retrying
        return ChatError("The model produced an invalid tool call.", retryable=True)
    if status in (401, 403) or name == "AuthenticationError":
        return ChatError("Authentication failed. Check that your API key is correct and active.")
    if status == 404 or name == "NotFoundError":
        return ChatError(f"Model or endpoint not found. Provider said: {detail}")
    if status == 429 or name == "RateLimitError":
        return ChatError("Rate limit reached. Please wait a moment and try again.", retryable=True)
    if status is not None and status >= 500:
        return ChatError("The AI service is having problems. Please try again shortly.", retryable=True)
    if status == 400 or name == "BadRequestError":
        return ChatError(f"The request was rejected. Provider said: {detail}")
    if name in {"APIConnectionError", "APITimeoutError", "Timeout"} or isinstance(exc, (ConnectionError, TimeoutError)):
        return ChatError("Could not reach the AI service (network problem or timeout).", retryable=True)
    return ChatError("Something went wrong while contacting the AI service.")


class LLMClient:
    """transport(messages, tools, temperature, max_tokens) -> {"content": str|None, "tool_calls": [...]}"""

    def __init__(self, config: Optional[LLMConfig] = None, transport: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self.model = config.model if config else "custom-transport"
        self.max_retries = config.max_retries if config else 3
        self._sleep = sleep
        self._transport = transport or self._openai_transport(config)

    @staticmethod
    def _openai_transport(cfg: LLMConfig):
        from openai import OpenAI
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout, max_retries=0)

        def call(messages, tools, temperature, max_tokens):
            kwargs = dict(model=cfg.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            if tools:
                kwargs["tools"] = tools
            msg = client.chat.completions.create(**kwargs).choices[0].message
            return {"content": msg.content,
                    "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                                   for tc in (msg.tool_calls or [])]}
        return call

    def chat(self, messages: list, tools: Optional[list] = None, temperature: float = 0.1,
             max_tokens: int = 700) -> LLMReply:
        last: Optional[ChatError] = None
        for attempt in range(self.max_retries + 1):
            try:
                out = self._transport(messages, tools, temperature, max_tokens)
                return LLMReply(out.get("content"), [ToolCall(t["id"], t["name"], t.get("arguments") or "{}")
                                                     for t in out.get("tool_calls", [])])
            except Exception as exc:  # noqa: BLE001
                last = classify_error(exc)
                if not last.retryable or attempt == self.max_retries:
                    raise last from exc
                self._sleep(min(8.0, 2 ** attempt + random.uniform(0, 0.5)))
        raise last  # pragma: no cover
