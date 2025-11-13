from abc import ABC, abstractmethod
import os
from openai import OpenAI
import tiktoken

# ------------------------
# Abstract Base Class
# ------------------------

class BaseLLM(ABC):
    def __init__(self, model_id: str, api_key: str, temperature: float = 0.0):
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def query_text(self, chat: list[dict[str, str]], max_tokens: int = 6000) -> str:
        pass

    @abstractmethod
    def query_chat(self, chat: list[dict[str, str]], max_completion_tokens: int = 6000) -> str:
        pass

    def count_tokens(self, text: str) -> int:
        """
        Counts tokens with cl100k_base (good baseline tokenizer for GPT-style models).
        """
        return len(self.tokenizer.encode(text))
    

class OpenAILLM(BaseLLM):
    def __init__(self, model_id: str, api_key: str, temperature: float = 0.0):
        self.client = OpenAI(api_key=api_key)
        try:
            self.encoder = tiktoken.encoding_for_model(model_id)
        except KeyError:
            print(f"[WARN] Unknown model '{model_id}', using cl100k_base tokenizer instead.")
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
        super().__init__(model_id, api_key, temperature)

    def list_models(self) -> list[str]:
        models = self.client.models.list()
        return [m.id for m in models.data]

    def query_text(self, chat: list[dict[str, str]], max_tokens: int = 1000) -> str:
        # --- Encode messages to estimate token count ---
        all_text = "\n".join(m["content"] for m in chat if "content" in m)
        tokens = self.encoder.encode(all_text)

        # --- Truncate if longer than 6000 tokens ---
        if len(tokens) > 6000:
            print(f"[WARN] Truncating input from {len(tokens)} → 6000 tokens")
            truncated_text = self.encoder.decode(tokens[-6000:])  # keep last 6000
            # replace only the last user message to fit in limit
            chat[-1]["content"] = truncated_text

        # for newer models take max_completion_tokens
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=chat,
            temperature=self.temperature,
            max_tokens=max_tokens, #, only for text models text model
            top_p=0
        )
        return response.choices[0].message.content
    
    def query_chat(self, chat: list[dict[str, str]], max_completion_tokens: int = 6000) -> str:
        """
        Query a chat-based model (e.g., gpt-4.1, gpt-5-nano).
        Uses the new chat.completions endpoint with max_completion_tokens.
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=chat,
            temperature=1.0,
            top_p=1.0
        )
        return response.choices[0].message.content
        


def load_openai_api_key(filepath=".openai_api_key") -> str:
    """Safely load an OpenAI API key from a local file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"API key file not found at {filepath}")
    with open(filepath, "r") as f:
        key = f.read().strip()
    if not key.startswith("sk-"):
        raise ValueError("Invalid API key format.")
    return key