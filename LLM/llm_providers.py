# llm_providers.py
import logging
from typing import Dict, Optional

from LLM.llmInterface import LLMInterface
from LLM.provider.anthropicLLM import AnthropicLLM
from LLM.provider.geminiLLM import GeminiLLM
from LLM.provider.huggingFaceLLM import HuggingFaceLLM
from LLM.provider.ollamaLM import OllamaLLM
from LLM.provider.openAILLM import OpenAILLM
from config import Config

logger = logging.getLogger(__name__)

# Sabitler
CONTENT_TYPE_JSON = "application/json"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
# Config dosyanıza DEFAULT_SYSTEM_MESSAGE ekleyebilirsiniz.
# Örn: Config.DEFAULT_SYSTEM_MESSAGE = "Sen yardımcı bir asistansın."
DEFAULT_SYSTEM_MESSAGE = getattr(Config, "DEFAULT_SYSTEM_MESSAGE", "Sen yardımcı bir asistansın.")


class LLMConfigurationError(Exception):
    """LLM yapılandırma hataları için özel exception."""
    pass


class LLMAPIError(Exception):
    """LLM API çağrı hataları için özel exception."""

    def __init__(self, message, provider: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self):
        details = f"Provider: {self.provider}"
        if self.status_code:
            details += f", Status Code: {self.status_code}"
        if self.response_text:
            # Yanıt çok uzunsa kısaltarak logla
            response_preview = (self.response_text[:200] + '...') if len(
                self.response_text) > 200 else self.response_text
            details += f", Response: {response_preview}"
        return f"{super().__str__()} ({details})"


class LLMProviderName:  # type: ignore
    """LLM sağlayıcı adları için sabitler."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LLMFactory:
    """LLM sağlayıcılarını oluşturan fabrika sınıfı."""

    _providers: Dict[str, type[LLMInterface]] = {
        LLMProviderName.OPENAI: OpenAILLM,
        LLMProviderName.ANTHROPIC: AnthropicLLM,
        LLMProviderName.GEMINI: GeminiLLM,
        LLMProviderName.HUGGINGFACE: HuggingFaceLLM,
        LLMProviderName.OLLAMA: OllamaLLM
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMInterface]):
        """Yeni bir LLM sağlayıcısı kaydeder."""
        normalized_name = name.lower()
        if normalized_name in cls._providers:
            logger.warning(f"Sağlayıcı '{normalized_name}' zaten kayıtlı. Üzerine yazılıyor.")
        cls._providers[normalized_name] = provider_class

    @classmethod
    def create_llm(cls, provider: Optional[str] = None, **kwargs) -> LLMInterface:
        """
        İstenilen LLM sağlayıcısını oluşturur.
        kwargs şu anda __init__ metodlarına geçirilmiyor.
        """
        provider_name_to_create = (provider or Config.DEFAULT_LLM_PROVIDER).lower()

        llm_class = cls._providers.get(provider_name_to_create)

        if llm_class:
            try:
                return llm_class()
            except LLMConfigurationError as e:  # __init__ içinde temel kontrollerden hata gelebilir
                logger.error(f"'{provider_name_to_create}' sağlayıcısı oluşturulurken yapılandırma hatası: {e}")
                raise  # Hatayı tekrar fırlat
            except Exception as e:
                logger.error(f"'{provider_name_to_create}' sağlayıcısı oluşturulurken beklenmedik hata: {e}",
                             exc_info=True)
                raise LLMConfigurationError(f"'{provider_name_to_create}' sağlayıcısı oluşturulamadı: {e}") from e

        else:
            available_providers = ", ".join(cls._providers.keys())
            error_message = (
                f"LLM sağlayıcısı '{provider_name_to_create}' bulunamadı. "
                f"Kullanılabilir sağlayıcılar: {available_providers}. "
                f"Lütfen Config.DEFAULT_LLM_PROVIDER veya `create_llm` çağrısını kontrol edin."
            )
            logger.error(error_message)
            raise ValueError(error_message)
