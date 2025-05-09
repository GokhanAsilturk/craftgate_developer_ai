from typing import Dict, Any, Optional

from LLM.llmInterface import LLMInterface
from LLM.llm_providers import LLMProviderName, LLMAPIError, logger, LLMConfigurationError
from config import Config


class HuggingFaceLLM(LLMInterface):
    def __init__(self):
        super().__init__(LLMProviderName.HUGGINGFACE)

    # HuggingFace için API URL'si provider adını içerdiğinden, generate_answer'da veya burada oluşturulmalı.
    # _get_config_value'dan base_url'i alıp burada birleştirmek daha temiz.
    def _get_hf_api_url(self, **kwargs) -> str:
        base_url = Config.API_URLS.get(self.provider_name)
        # Model, kwargs'tan (generate_answer'a özel) veya config'den (varsayılan) alınabilir.
        model_id = kwargs.get("") or Config.LLM_MODELS.get(self.provider_name)

        if not base_url or not model_id:
            raise LLMConfigurationError(
                f"{self.provider_name.capitalize()} için API base URL veya provider ID eksik. "
                "Lütfen Config.API_URLS ve Config.LLM_MODELS'i kontrol edin."
            )
        return f"{base_url.rstrip('/')}/{model_id}"

    # generate_answer'ı override ederek URL'yi özel olarak ayarlayalım
    def generate_answer(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        try:
            api_url = self._get_hf_api_url(**kwargs)  # HF'ye özel URL alma
            payload = self._prepare_payload(question, context, **kwargs)
            headers = self._prepare_headers(**kwargs)

            response = self._make_request("POST", api_url, headers=headers, json_payload=payload, **kwargs)
            response_data = response.json()
            return self._parse_response(response_data)

        except LLMConfigurationError:
            raise
        except LLMAPIError:
            raise
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"{self.provider_name.capitalize()} yanıtını işlerken beklenmedik format: {e}", exc_info=True)
            raise LLMAPIError(message="API yanıtı beklenmedik bir formattaydı.", provider=self.provider_name) from e
        except Exception as e:
            logger.error(f"{self.provider_name.capitalize()} yanıt üretirken genel bir hata oluştu: {e}", exc_info=True)
            raise LLMAPIError(message=f"Beklenmedik bir hata oluştu: {e}", provider=self.provider_name) from e

    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        prompt_content = f"{context}\n\nSoru: {question}" if context else question
        payload: Dict[str, Any] = {"inputs": prompt_content}

        hf_params = {}
        # HF için sıcaklık genellikle 0.0-100.0 arası veya 0.0-1.0 olabilir, modele göre değişir.
        # Biz 0.0-1.0 aralığını varsayıyoruz ve 0 olmamasını sağlıyoruz.
        temperature_hf = self._get_config_value("temperature", kwargs.get("temperature"))
        if temperature_hf is not None:
            hf_params["temperature"] = max(0.01, float(temperature_hf))  # 0 olmamalı

        max_tokens_hf = self._get_config_value("max_tokens", kwargs.get("max_tokens"))
        if max_tokens_hf is not None:  # HF'de "max_new_tokens" veya "max_length" olabilir
            hf_params["max_new_tokens"] = int(max_tokens_hf)

        # Diğer yaygın HF parametreleri
        hf_params.update({k: v for k, v in kwargs.items() if k in [
            "top_k", "top_p", "repetition_penalty", "max_time", "num_return_sequences"
        ] and v is not None})

        if hf_params:
            payload["parameters"] = hf_params

        payload.setdefault("options", {}).update({
            "wait_for_model": kwargs.get("wait_for_model", True),
            "use_cache": kwargs.get("use_cache", True)  # Önbellek kullanımı genellikle iyidir
        })
        return payload

    def _parse_response(self, response_data: Any) -> str:
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
