"""Local-only Ollama structured generation without an additional SDK."""

from urllib.parse import urlsplit

import requests

DEFAULT_MODEL = "gemma4:12b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


def local_base_url(value: str) -> str:
    """Do not accidentally send community data to a remote host or a proxy."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP(S) URL") from None
    if (
        parsed.scheme not in ("http", "https")
        or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or port == 0
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP(S) URL without credentials or a path")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"{parsed.scheme}://{host}" + (f":{port}" if port else "")


class OllamaClient:
    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, session=None):
        self.base_url = local_base_url(base_url)
        self.session = session or requests.Session()
        self.session.trust_env = False

    def generate(self, *, model: str, instructions: str, prompt: str, schema: dict) -> str:
        if not isinstance(model, str) or not model.strip() or any(ord(c) < 32 for c in model):
            raise ValueError("A local Ollama model name is required")
        if model.lower().endswith((":cloud", "-cloud")):
            raise ValueError("Choose an installed local model, not an Ollama cloud model")
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "format": schema,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0, "num_predict": 5000},
                },
                timeout=(3.05, 120),
                allow_redirects=False,
            )
        except requests.Timeout:
            raise OllamaError("The local model timed out; try a shorter trip or smaller model.") from None
        except requests.RequestException:
            raise OllamaError("Cannot reach local Ollama. Start it with ollama serve.") from None
        if response.status_code == 404:
            raise OllamaError("The selected local model is unavailable. Check ollama list and OLLAMA_MODEL.")
        if response.status_code != 200:
            raise OllamaError("Local Ollama could not generate a draft.")
        try:
            data = response.json()
        except ValueError:
            raise OllamaError("Local Ollama returned invalid JSON.") from None
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip() or data.get("done") is not True:
            raise OllamaError("Local Ollama returned an incomplete draft.")
        if data.get("done_reason") == "length":
            raise OllamaError("The local draft exceeded its output limit; try a shorter trip.")
        return content
