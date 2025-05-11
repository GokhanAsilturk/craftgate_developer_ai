import os
import textwrap
from datetime import datetime
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

        try:
            # Kayıt için bir dizin oluşturun (varsa atla)
            dump_dir = "prompt_dumps"
            os.makedirs(dump_dir, exist_ok=True)

            # Dosya adı için zaman damgası kullanın
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(dump_dir, f"{self.provider_name}_prompt_{timestamp}.txt")

            # Prompt içeriğini dosyaya yazın
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Provider: {self.provider_name}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write("-" * 20 + "\n")

                if system_instruction_content:
                    f.write("System Message:\n")
                    wrapped_system_instruction = textwrap.fill(system_instruction_content,
                                                               width=80)  # 80 karakter genişliğinde satırlar
                    f.write(wrapped_system_instruction + "\n")
                    f.write("-" * 20 + "\n")

                f.write("Prompt Content (Context + Question):\n")
                wrapped_prompt_content = textwrap.fill(prompt_content, width=80)
                f.write(wrapped_prompt_content + "\n")
                f.write("=" * 40 + "\n")

            logger.info(f"Prompt içeriği '{filename}' dosyasına kaydedildi.")

        except Exception as e:
            logger.error(f"Prompt içeriğini kaydederken hata oluştu: {e}")

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
