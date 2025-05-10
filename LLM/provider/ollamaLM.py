import json
from typing import Optional, Dict, Any

import requests

from LLM.llmInterface import LLMInterface
from LLM.llm_constants import LLMProviderName, LLMAPIError, logger, LLMConfigurationError, DEFAULT_SYSTEM_MESSAGE
from config import Config


class OllamaLLM(LLMInterface):
    def __init__(self):
        super().__init__(LLMProviderName.OLLAMA)

    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        model = self.get_config_value(kwargs, "provider", Config)
        system_message_content = kwargs.get('system_message', DEFAULT_SYSTEM_MESSAGE)
        # Ollama'ya özel 'stream' Config.MODEL_SPECIFIC_PARAMS'tan veya kwargs'tan gelebilir
        stream = self.get_config_value(kwargs, "stream", Config, default_value=False)

        prompt_content = f"{context}\n\nSoru: {question}" if context else question
        payload: Dict[str, Any] = {
            "provider": model,
            "prompt": prompt_content,
            "stream": stream
        }
        if system_message_content:
            payload["system"] = system_message_content

        ollama_options = {}
        common_params = self._prepare_common_payload_params(**kwargs)  # temperature
        if "temperature" in common_params: ollama_options["temperature"] = common_params["temperature"]
        # max_tokens Ollama'da "num_predict" olarak geçer
        max_tokens_ollama = self.get_config_value("max_tokens", kwargs.get("max_tokens"))
        if max_tokens_ollama is not None: ollama_options["num_predict"] = int(max_tokens_ollama)

        # Diğer Ollama'ya özel 'options' parametreleri
        ollama_options.update({k: v for k, v in kwargs.items() if k in [
            "top_k", "top_p", "seed", "stop", "num_ctx", "num_gpu", "mirostat", "mirostat_eta", "mirostat_tau"
            # num_gpu gibi sayısal değerler için tip dönüşümü gerekebilir.
        ] and v is not None})

        if ollama_options:
            payload["options"] = ollama_options
        return payload

    def _parse_response(self, response_data: Any) -> str:
        # Stream olmayan yanıt için:
        if isinstance(response_data, dict) and "response" in response_data:
            return response_data["response"]
        # Stream olan yanıt _make_request içinde ele alınmalı veya burada response objesi direkt işlenmeli.
        # Şimdiki yapıda _make_request response.json() döndürdüğü için stream burada tam işlenemez.
        # Stream'i desteklemek için _make_request'i ve generate_answer'ı Ollama için override etmek gerekebilir.
        # VEYA stream=True ise _make_request response objesini döndürür, burada işlenir.
        # Şimdilik stream olmayan yanıta odaklanalım.
        if isinstance(response_data, dict) and "error" in response_data:
            error_msg = response_data["error"]
            logger.error(f"Ollama API hatası: {error_msg}")
            raise LLMAPIError(message=f"API hatası: {error_msg}", provider=self.provider_name,
                              response_text=str(response_data))

        logger.warning(
            f"{self.provider_name.capitalize()} API'den beklenmedik yanıt formatı (stream olmayan): {response_data}")
        raise LLMAPIError(message="Yanıt formatı ayrıştırılamadı.", provider=self.provider_name,
                          response_text=str(response_data))

    # Ollama stream yanıtını düzgün işlemek için generate_answer'ı override edelim.
    def generate_answer(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        try:
            api_url = self.get_config_value(kwargs, "api_url", Config)
            payload = self._prepare_payload(question, context, **kwargs)
            headers = self._prepare_headers(**kwargs)  # Ollama için genellikle özel header gerekmez

            if payload.get("stream"):
                # Stream için farklı istek ve yanıt işleme
                response = self._make_request("POST", api_url, headers=headers, json_payload=payload, stream=True,
                                              **kwargs)  # _make_request'e stream=True ilet

                full_response_content = ""
                last_json_part = {}
                try:
                    for line in response.iter_lines():  # iter_lines byte döndürür
                        if line:
                            decoded_line = line.decode('utf-8')
                            json_part = json.loads(decoded_line)
                            last_json_part = json_part
                            full_response_content += json_part.get("response", "")
                            if json_part.get("done") and json_part.get("error"):  # Stream ortasında hata
                                error_msg = json_part.get("error")
                                logger.error(f"Ollama stream API hatası: {error_msg}")
                                raise LLMAPIError(message=f"Stream sırasında API hatası: {error_msg}",
                                                  provider=self.provider_name)

                    if last_json_part.get("done"):
                        if not full_response_content and last_json_part.get("error"):  # Stream sonunda genel hata
                            raise LLMAPIError(message=f"Stream API hatası: {last_json_part.get('error')}",
                                              provider=self.provider_name)
                        return full_response_content
                    else:  # Stream beklenmedik şekilde bitti
                        raise LLMAPIError(message="Stream yanıtı tamamlanmadı.", provider=self.provider_name)

                except requests.exceptions.RequestException as e:  # Bağlantı hatası stream sırasında
                    logger.error(f"Ollama stream bağlantı hatası: {e}", exc_info=True)
                    raise LLMAPIError(message=f"Stream API'ye bağlanırken bir sorun oluştu: {e}",
                                      provider=self.provider_name) from e
                except json.JSONDecodeError as e:
                    logger.error(f"Ollama stream yanıtı JSON olarak çözümlenemedi: {e}", exc_info=True)
                    raise LLMAPIError(message="Stream yanıtı geçersiz JSON formatındaydı.",
                                      provider=self.provider_name) from e
            else:
                # Stream olmayan normal istek
                response = self._make_request("POST", api_url, headers=headers, json_payload=payload, **kwargs)
                response_data = response.json()
                return self._parse_response(response_data)

        except LLMConfigurationError:
            raise
        except LLMAPIError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name.capitalize()} yanıt üretirken genel bir hata oluştu: {e}", exc_info=True)
            raise LLMAPIError(message=f"Beklenmedik bir hata oluştu: {e}", provider=self.provider_name) from e

    # _make_request'i stream parametresini alacak şekilde güncelleyelim (Ollama için)
    def _make_request(self, method: str, url: str, headers: Dict[str, str],
                      json_payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None,
                      stream: bool = False, **kwargs) -> requests.Response:
        """Merkezi HTTP istek fonksiyonu (stream destekli)."""
        timeout = self.get_config_value(kwargs, "timeout", Config)
        try:
            # stream=True ise requests.post stream parametresini alır
            response = requests.request(method, url, headers=headers, json=json_payload, params=params, timeout=timeout,
                                        stream=stream)
            if not stream:  # Stream olmayan yanıtlar için HTTP hata kontrolü
                response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            response_text = None
            try:
                response_text = e.response.json()
            except:
                response_text = e.response.text
            logger.error(
                f"{self.provider_name.capitalize()} API HTTP Hatası: {e.response.status_code} - {response_text}",
                exc_info=True)
            raise LLMAPIError(message="API çağrısı başarısız oldu.", provider=self.provider_name,
                              status_code=e.response.status_code, response_text=str(response_text)) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"{self.provider_name.capitalize()} API Bağlantı Hatası: {e}", exc_info=True)
            raise LLMAPIError(message=f"API'ye bağlanırken bir sorun oluştu: {e}", provider=self.provider_name) from e
