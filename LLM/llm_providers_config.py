# LLM/llm_providers_config.py
from typing import Type

import requests

from LLM.llmInterface import LLMInterface
from LLM.llm_constants import (
    logger, CONTENT_TYPE_JSON, ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM,
    DEFAULT_SYSTEM_MESSAGE, LLMProviderName, LLMConfigurationError
)
from LLM.provider.anthropicLLM import AnthropicLLM
from LLM.provider.geminiLLM import GeminiLLM
from LLM.provider.huggingFaceLLM import HuggingFaceLLM
from LLM.provider.ollamaLM import OllamaLLM
from config import Config


class OpenAILLM(LLMInterface):
    """OpenAI (GPT) API'sini kullanan LLM sağlayıcısı."""

    def __init__(self):
        super().__init__(LLMProviderName.OPENAI.value)

    def generate_answer(self, question: str, context: str = None, **kwargs) -> str:
        """OpenAI API'sini kullanarak yanıt üretir."""
        try:
            api_url = self.get_config_value(kwargs, "api_url", Config)
            api_key = self.get_config_value(kwargs, "api_key", Config)
            model = self.get_config_value(kwargs, "model", Config)
            temperature = self.get_config_value(kwargs, "temperature", Config)
            max_tokens = self.get_config_value(kwargs, "max_tokens", Config)

            headers = {
                "Content-Type": CONTENT_TYPE_JSON,
                "Authorization": f"Bearer {api_key}"
            }

            # Mesaj listesini hazırla
            messages = []

            # Sistem mesajını ekle (eğer varsa)
            system_message = kwargs.get('system_message', DEFAULT_SYSTEM_MESSAGE)
            messages.append({"role": ROLE_SYSTEM, "content": system_message})

            # Bağlam bilgisini ekle
            if context:
                messages.append(
                    {"role": ROLE_USER, "content": f"Aşağıdaki bilgileri kullanarak soru yanıtla:\n\n{context}"})
                messages.append({"role": ROLE_ASSISTANT,
                                 "content": "Anlaşıldı, bu bilgileri kullanarak sorunuzu yanıtlayabilirim."})

            # Soruyu ekle
            messages.append({"role": ROLE_USER, "content": question})

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"OpenAI API hatası: {str(e)}")
            return self.get_error_message()


class LLMFactory:
    """LLM sağlayıcılarını oluşturan fabrika sınıfı."""

    _providers = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_class: Type[LLMInterface]):
        """
        Yeni bir LLM sağlayıcısını kaydeder.

        Args:
            provider_name: Sağlayıcı adı
            provider_class: Sağlayıcı sınıfı (LLMInterface alt sınıfı)
        """
        if not issubclass(provider_class, LLMInterface):
            raise LLMConfigurationError(f"{provider_class.__name__} sınıfı LLMInterface'den türetilmelidir")

        cls._providers[provider_name.lower()] = provider_class

    @classmethod
    def create_llm(cls, provider: str = None, **kwargs) -> LLMInterface:
        # ...
        if not cls._providers:
            cls.register_provider(LLMProviderName.OPENAI.value, OpenAILLM)
            cls.register_provider(LLMProviderName.GEMINI.value, GeminiLLM)
            cls.register_provider(LLMProviderName.ANTHROPIC.value, AnthropicLLM)
            cls.register_provider(LLMProviderName.OLLAMA.value, OllamaLLM)
            cls.register_provider(LLMProviderName.HUGGINGFACE.value, HuggingFaceLLM)

        if provider is None:
            provider = Config.DEFAULT_LLM_PROVIDER.lower()
        else:
            provider = provider.lower()

        if provider in cls._providers:
            # İstenen sağlayıcı kayıtlıysa onu döndür
            logger.info(f"LLM sağlayıcısı kullanılıyor: {provider}")  # Hangi sağlayıcının kullanıldığını logla
            return cls._providers[provider]()
        else:
            # Eğer istenen sağlayıcı hala bulunamazsa (bu durumda bir sorun var demektir)
            # bir hata fırlatmak daha iyi olabilir, ya da gerçekten bir varsayılan
            error_msg = f"LLM sağlayıcısı '{provider}' kayıtlı değil. Lütfen 'llm_providers_config.py' dosyasını kontrol edin."
            logger.error(error_msg)
            # Alternatif olarak, eğer bir fallback isteniyorsa:
            # logger.warning(f"Bilinmeyen LLM sağlayıcısı: {provider}. Varsayılan olarak OpenAI kullanılıyor.")
            # return cls._providers[LLMProviderName.OPENAI.value]()
            raise LLMConfigurationError(error_msg)  # Net bir hata fırlatmak daha iyi
