import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from fastapi import requests

from LLM.llm_providers import LLMProviderName, LLMConfigurationError, logger, LLMAPIError, CONTENT_TYPE_JSON
from config import Config


class LLMInterface(ABC):
    """Tüm LLM sağlayıcıları için temel arayüz."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name.lower()
        # Temel yapılandırma kontrolü
        if self.provider_name not in Config.LLM_MODELS:
            # Ollama gibi bazıları LLM_MODELS'de olmayabilir ama API_URLS'de olmalı
            if self.provider_name not in Config.API_URLS:
                logger.warning(
                    f"'{self.provider_name}' için temel yapılandırma (LLM_MODELS veya API_URLS) "
                    f"Config dosyasında eksik olabilir."
                )

    def _get_config_value(self, key: str, kwargs_value: Optional[Any] = None,
                          default_value: Optional[Any] = None) -> Any:
        """
        Yapılandırma değerini alır. Öncelik sırası:
        1. kwargs ile fonksiyona direkt verilen değer.
        2. Config.MODEL_SPECIFIC_PARAMS[provider_name].get(key)
        3. Anahtar türüne göre Config'den (API_URLS, API_KEYS, LLM_MODELS).
        4. Config.LLM_PARAMS.get(key) (genel parametreler için).
        5. Metoda gönderilen `default_value`.
        """
        if kwargs_value is not None:
            return kwargs_value

        provider_specific_config = Config.MODEL_SPECIFIC_PARAMS.get(self.provider_name, {})
        if key in provider_specific_config:
            return provider_specific_config[key]

        config_sources = {
            "api_url": Config.API_URLS,
            "api_key": Config.API_KEYS,  # Ortam değişkeni mantığı aşağıda ayrıca ele alınacak
            "provider": Config.LLM_MODELS,
        }

        if key in config_sources:
            source_dict = config_sources[key]
            # HuggingFace için özel URL birleştirme (provider adı API_URLS'de yoksa)
            if key == "api_url" and self.provider_name == LLMProviderName.HUGGINGFACE:
                base_url = source_dict.get(self.provider_name)
                # model_for_hf, generate_answer'dan kwargs ile veya _get_config_value('provider') ile alınır
                # Bu metot içinde direkt kwargs'a erişim yok, çağıran yer modeli sağlamalı
                # Bu yüzden HF URL birleştirmesini HFLLM sınıfına taşıyacağız.
                # Şimdilik base_url'i döndürsün.
                if base_url: return base_url
            elif key == "api_key" and self.provider_name != LLMProviderName.OLLAMA:
                # API anahtarını Config.API_KEYS'ten almayı dene
                api_key_from_config = source_dict.get(self.provider_name)
                env_var_name = f"{self.provider_name.upper()}_API_KEY"
                api_key_from_env = os.getenv(env_var_name)

                if api_key_from_env:  # Ortam değişkeni öncelikli
                    return api_key_from_env
                if api_key_from_config and "your_" not in str(
                        api_key_from_config).lower() and api_key_from_config:  # Config'de geçerli anahtar
                    return api_key_from_config
                if api_key_from_config:  # Config'de placeholder varsa
                    logger.warning(
                        f"'{self.provider_name}' için API anahtarı Config.API_KEYS'te placeholder ('{api_key_from_config}') "
                        f"olarak ayarlanmış ve {env_var_name} ortam değişkeni bulunamadı."
                    )
                    # Placeholder'ı döndürmek yerine hata fırlatmak daha güvenli olabilir.
                    # Şimdilik placeholder'ı döndürelim, API çağrısı başarısız olacaktır.
                    return api_key_from_config
                # Ne config'de ne de env'de yoksa, hata aşağıda fırlatılacak.

            value = source_dict.get(self.provider_name)
            if value is not None:
                return value

        if key in Config.LLM_PARAMS:
            return Config.LLM_PARAMS[key]

        if default_value is not None:
            return default_value

        # Zorunlu alanlar için kontrol (Ollama API key gerektirmez)
        if key == "api_url" or \
                (key == "api_key" and self.provider_name != LLMProviderName.OLLAMA) or \
                (
                        key == "provider" and self.provider_name != LLMProviderName.GEMINI):  # Gemini provider adını URL'den alır
            raise LLMConfigurationError(
                f"'{self.provider_name}' için zorunlu yapılandırma değeri '{key}' bulunamadı. "
                f"Lütfen config.py dosyasını ve ilgili ortam değişkenlerini kontrol edin (örn: {self.provider_name.upper()}_API_KEY)."
            )

        logger.debug(
            f"'{self.provider_name}' için '{key}' yapılandırma değeri bulunamadı, None/default_value kullanılıyor.")
        return None

    def _prepare_common_payload_params(self, **kwargs) -> Dict[str, Any]:
        """kwargs ve Config'den genel LLM parametrelerini alır."""
        params = {}
        temperature = self._get_config_value("temperature", kwargs.get("temperature"))
        if temperature is not None: params["temperature"] = temperature

        max_tokens = self._get_config_value("max_tokens", kwargs.get("max_tokens"))
        if max_tokens is not None: params["max_tokens"] = max_tokens

        # Diğer bilinen ortak parametreler eklenebilir (örn: top_p, top_k)
        # Ancak bu parametrelerin adları API'den API'ye değişebilir (örn: max_tokens vs maxOutputTokens)
        # Bu yüzden bunları her LLM sınıfında özel olarak ele almak daha iyi olabilir.
        return params

    def _make_request(self, method: str, url: str, headers: Dict[str, str],
                      json_payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None,
                      **kwargs) -> requests.Response:
        """Merkezi HTTP istek fonksiyonu."""
        timeout = self._get_config_value("timeout", kwargs.get("timeout"))
        try:
            response = requests.request(method, url, headers=headers, json=json_payload, params=params, timeout=timeout)
            response.raise_for_status()  # HTTP 4xx/5xx hataları için exception fırlatır
            return response
        except requests.exceptions.HTTPError as e:
            # Detaylı hata mesajı için yanıt içeriğini logla/kullan
            response_text = None
            try:
                response_text = e.response.json()
            except requests.exceptions.JSONDecodeError:
                response_text = e.response.text
            logger.error(
                f"{self.provider_name.capitalize()} API HTTP Hatası: {e.response.status_code} - {response_text}",
                exc_info=True)
            raise LLMAPIError(
                message=f"API çağrısı başarısız oldu.",
                provider=self.provider_name,
                status_code=e.response.status_code,
                response_text=str(response_text)
            ) from e
        except requests.exceptions.RequestException as e:  # Bağlantı hataları, timeout vb.
            logger.error(f"{self.provider_name.capitalize()} API Bağlantı Hatası: {e}", exc_info=True)
            raise LLMAPIError(
                message=f"API'ye bağlanırken bir sorun oluştu: {e}",
                provider=self.provider_name
            ) from e

    @abstractmethod
    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Her sağlayıcı için API isteğinin gövdesini hazırlar."""
        pass

    @abstractmethod
    def _parse_response(self, response_data: Any) -> str:
        """Her sağlayıcının API yanıtını ayrıştırır ve metin yanıtını döndürür."""
        pass

    def generate_answer(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Verilen soru ve bağlam üzerinden yanıt üretir.
        Bu metot artık Template Method Paternini daha net kullanıyor.
        """
        try:
            api_url = self._get_config_value("api_url", kwargs.get("api_url"))
            # API anahtarı ve diğer başlıklar _prepare_headers içinde ele alınacak

            payload = self._prepare_payload(question, context, **kwargs)
            headers = self._prepare_headers(**kwargs)

            # Gemini gibi bazı API'ler query parametresi olarak key alabilir.
            query_params = self._get_query_params(**kwargs)

            response = self._make_request("POST", api_url, headers=headers, json_payload=payload, params=query_params,
                                          **kwargs)
            response_data = response.json()
            return self._parse_response(response_data)

        except LLMConfigurationError:  # _get_config_value'dan gelebilir
            raise  # Olduğu gibi tekrar fırlat
        except LLMAPIError:  # _make_request'ten veya _parse_response'tan gelebilir
            raise  # Olduğu gibi tekrar fırlat
        except (KeyError, IndexError, TypeError) as e:  # Yanıt ayrıştırma sırasında beklenmedik format
            logger.error(f"{self.provider_name.capitalize()} yanıtını işlerken beklenmedik format: {e}", exc_info=True)
            raise LLMAPIError(message="API yanıtı beklenmedik bir formattaydı.", provider=self.provider_name) from e
        except Exception as e:  # Diğer beklenmedik hatalar
            logger.error(f"{self.provider_name.capitalize()} yanıt üretirken genel bir hata oluştu: {e}", exc_info=True)
            raise LLMAPIError(message=f"Beklenmedik bir hata oluştu: {e}", provider=self.provider_name) from e

    def _prepare_headers(self, **kwargs) -> Dict[str, str]:
        """Genel başlıkları ve sağlayıcıya özel başlıkları hazırlar."""
        headers = {"Content-Type": CONTENT_TYPE_JSON}
        api_key_val = self._get_config_value("api_key", kwargs.get("api_key"))

        if self.provider_name == LLMProviderName.OPENAI:
            headers["Authorization"] = f"Bearer {api_key_val}"
        elif self.provider_name == LLMProviderName.ANTHROPIC:
            headers["X-API-Key"] = api_key_val
            # Anthropic version config'den veya kwargs'tan alınabilir
            anthropic_version = self._get_config_value(
                "anthropic_version",
                kwargs.get("anthropic_version"),
                default_value=Config.MODEL_SPECIFIC_PARAMS.get(LLMProviderName.ANTHROPIC, {}).get("anthropic_version",
                                                                                                  "2023-06-01")
            )
            headers["anthropic-version"] = anthropic_version
        elif self.provider_name == LLMProviderName.HUGGINGFACE:
            headers["Authorization"] = f"Bearer {api_key_val}"
        # Gemini ve Ollama için özel başlık gerekmiyorsa burası boş kalabilir.
        # Gemini API anahtarını query parametresi olarak alır.
        return headers

    def _get_query_params(self, **kwargs) -> Optional[Dict[str, str]]:
        """Bazı API'ler için query parametrelerini hazırlar (örn: Gemini)."""
        if self.provider_name == LLMProviderName.GEMINI:
            api_key_val = self._get_config_value("api_key", kwargs.get("api_key"))
            return {"key": api_key_val}
        return None
