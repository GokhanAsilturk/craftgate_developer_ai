# LLM/huggingFaceLLM.py

from typing import Dict, Any, Optional

from LLM.llmInterface import LLMInterface
from LLM.llm_constants import LLMProviderName, LLMAPIError, logger, LLMConfigurationError, CONTENT_TYPE_JSON
from config import Config


class HuggingFaceLLM(LLMInterface):
    def __init__(self):
        # LLMProviderName enum değerini __init__ metoduna gönderiyoruz
        super().__init__(LLMProviderName.HUGGINGFACE)

    # HuggingFace için API URL'si provider adını içerdiğinden,
    # generate_answer'ı override ederek URL'yi özel olarak ayarlayalım
    def generate_answer(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Hugging Face API'sini kullanarak yanıt üretir (URL'yi özel hazırlar)."""
        try:
            # URL'yi Config'den base URL'yi alıp modele göre oluşturun
            base_url = self.get_config_value(kwargs, "api_url", Config)  # Base URL'yi alın
            # Model adını Config'den veya kwargs'tan alın
            model_id = self.get_config_value(kwargs, "model", Config)

            if not base_url or not model_id:
                raise LLMConfigurationError(
                    f"{self.provider_name.capitalize()} için API base URL veya model ID eksik. "
                    "Lütfen Config.API_URLS ve Config.LLM_MODELS'i kontrol edin."
                )
            api_url = f"{base_url.rstrip('/')}/{model_id}"  # URL'yi oluşturun

            # Payload'ı hazırlayın (sağlayıcıya özel metod)
            payload = self._prepare_payload(question, context, **kwargs)

            # API Key'i Config'den veya kwargs'tan alın (generate_answer içinde de alınabilir)
            api_key = self.get_config_value(kwargs, "api_key", Config)

            # Headers'ı hazırlayın (API Key'i içeren)
            headers = {
                "Content-Type": CONTENT_TYPE_JSON,  # Genellikle JSON gönderilir
            }
            # API Key header'ı ekleyin
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # HTTP isteği yapın (LLMInterface'deki ortak metod)
            response = self._make_request("POST", api_url, headers=headers, json_payload=payload, **kwargs)

            # Yanıt verisini alın
            response_data = response.json()

            # Yanıtı ayrıştırın (sağlayıcıya özel metod)
            return self._parse_response(response_data)

        except LLMConfigurationError:
            # Config hatası zaten yakalandı ve loglandı/fırlatıldı, bunu tekrar fırlat
            raise
        except LLMAPIError:
            # _make_request veya _parse_response tarafından fırlatılan API hataları
            raise
        except (KeyError, IndexError, TypeError) as e:
            # Yanıtı işlerken (json parse veya key hatası) beklenmedik format hatası
            logger.error(f"{self.provider_name.capitalize()} yanıtını işlerken beklenmedik format: {e}", exc_info=True)
            raise LLMAPIError(message="API yanıtı beklenmedik bir formattaydı.", provider=self.provider_name) from e
        except Exception as e:
            # Beklenmeyen diğer hataları yakala
            logger.error(f"{self.provider_name.capitalize()} yanıt üretirken genel bir hata oluştu: {e}", exc_info=True)
            raise LLMAPIError(message=f"Beklenmedik bir hata oluştu: {e}", provider=self.provider_name) from e

    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Hugging Face API'si için payload'ı hazırlar."""
        # Config objesine get_config_value içinde erişilecek
        from config import Config  # Metod içinde import

        prompt_content = f"{context}\n\nSoru: {question}" if context else question
        payload: Dict[str, Any] = {"inputs": prompt_content}

        hf_params = {}
        # HF için sıcaklık genellikle 0.0-100.0 arası veya 0.0-1.0 olabilir, modele göre değişir.
        # Biz 0.0-1.0 aralığını varsayıyoruz ve 0 olmamasını sağlıyoruz.
        temperature_hf = self.get_config_value(kwargs, "temperature", Config)
        if temperature_hf is not None:
            hf_params["temperature"] = max(0.01, float(temperature_hf))  # 0 olmamalı

        max_tokens_hf = self.get_config_value(kwargs, "max_tokens", Config)
        if max_tokens_hf is not None:  # HF'de "max_new_tokens" veya "max_length" olabilir
            hf_params["max_new_tokens"] = int(max_tokens_hf)

        # Diğer yaygın HF parametreleri
        hf_params.update({k: v for k, v in kwargs.items() if k in [
            "top_k", "top_p", "repetition_penalty", "max_time", "num_return_sequences"
        ] and v is not None})

        if hf_params:
            payload["parameters"] = hf_params

        # Modele özel parametreleri doğrudan kwargs'tan alabiliriz veya config'den de bakabiliriz
        # Config.MODEL_SPECIFIC_PARAMS içinde tanımlanmış Hugging Face parametreleri varsa
        wait_for_model = self.get_config_value(kwargs, "wait_for_model", Config, default_value=True)
        use_cache = self.get_config_value(kwargs, "use_cache", Config, default_value=True)

        payload.setdefault("options", {}).update({
            "wait_for_model": wait_for_model,
            "use_cache": use_cache  # Önbellek kullanımı genellikle iyidir
        })
        return payload

    def _parse_response(self, response_data: Any) -> str:
        """Hugging Face API yanıtından metin yanıtını ayrıştırır."""
        if isinstance(response_data, list) and response_data:
            if "generated_text" in response_data[0]:
                return response_data[0]["generated_text"]
        elif isinstance(response_data, dict) and "generated_text" in response_data:
            return response_data["generated_text"]
        # Bazı HF modelleri hata durumunda {'error': 'mesaj', 'warnings': []} dönebilir
        if isinstance(response_data, dict) and "error" in response_data:
            error_msg = response_data["error"]
            warnings = response_data.get("warnings")
            logger.error(f"HuggingFace API hatası: {error_msg}, Uyarılar: {warnings}")
            raise LLMAPIError(message=f"API hatası: {error_msg}", provider=self.provider_name,
                              response_text=str(response_data))

        logger.warning(f"{self.provider_name.capitalize()} API'den beklenmedik yanıt formatı: {response_data}")
        raise LLMAPIError(message="Yanıt formatı ayrıştırılamadı.", provider=self.provider_name,
                          response_text=str(response_data))
