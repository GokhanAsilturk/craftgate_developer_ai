from typing import Dict, Any, Optional

from LLM.llmInterface import LLMInterface
from LLM.llm_constants import LLMProviderName, DEFAULT_SYSTEM_MESSAGE, logger, LLMAPIError
from config import Config


class GeminiLLM(LLMInterface):
    def __init__(self):
        super().__init__(LLMProviderName.GEMINI)
        # Gemini'nin provider adı URL'de belirtildiği için (Config.API_URLS["gemini"])
        # get_config_value("provider") çağrısı burada gereksiz.

    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        system_instruction_content = kwargs.get('system_message', DEFAULT_SYSTEM_MESSAGE)
        prompt_content = f"{context}\n\nSoru: {question}" if context else question

        contents = [{"parts": [{"text": prompt_content}]}]
        # TODO: Gemini için context ve chat geçmişi yönetimi daha gelişmiş bir "contents" yapısı gerektirebilir.

        payload: Dict[str, Any] = {"contents": contents}

        generation_config = {}
        temperature = self.get_config_value(kwargs, "temperature", Config)  # <-- Düzeltildi: kwargs, key, Config
        if temperature is not None: generation_config["temperature"] = temperature

        max_output_tokens = self.get_config_value(kwargs, "max_tokens",
                                                  Config)  # Gemini'de maxOutputTokens # <-- Düzeltildi

        if max_output_tokens is not None: generation_config["maxOutputTokens"] = max_output_tokens

        # Gemini'ye özel diğer generationConfig parametreleri
        generation_config.update(
            {k: v for k, v in kwargs.items() if k in ['topP', 'topK', 'candidateCount'] and v is not None})
        if generation_config:
            payload["generationConfig"] = generation_config

        if system_instruction_content:
            payload['system_instruction'] = {'parts': [{'text': system_instruction_content}]}
        return payload

    def _parse_response(self, response_data: Any) -> str:
        if response_data.get("candidates") and response_data["candidates"][0].get("content", {}).get("parts"):
            return response_data["candidates"][0]["content"]["parts"][0]["text"]

        # Güvenlik filtrelerini ve diğer hata durumlarını kontrol et
        if response_data.get("candidates") and response_data["candidates"][0].get("finishReason") == "SAFETY":
            safety_ratings = response_data["candidates"][0].get('safetyRatings', 'Detay yok')
            logger.error(f"{self.provider_name.capitalize()} yanıtı güvenlik filtrelerine takıldı: {safety_ratings}")
            raise LLMAPIError(
                message=f"İçerik güvenlik politikaları nedeniyle üretilemedi. Detay: {safety_ratings}",
                provider=self.provider_name,
                response_text=str(response_data)
            )
        if response_data.get("error"):
            error_details = response_data.get("error")
            logger.error(f"{self.provider_name.capitalize()} API hatası döndürdü: {error_details}")
            raise LLMAPIError(
                message=f"API hatası: {error_details.get('message', 'Bilinmeyen hata')}",
                provider=self.provider_name,
                status_code=error_details.get('code'),
                response_text=str(response_data)
            )
        logger.warning(
            f"{self.provider_name.capitalize()} API'den beklenmedik yanıt formatı veya boş yanıt: {response_data}")
        raise LLMAPIError(message="Yanıt formatı ayrıştırılamadı.", provider=self.provider_name,
                          response_text=str(response_data))
