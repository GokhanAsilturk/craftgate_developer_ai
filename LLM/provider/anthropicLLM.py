from typing import Optional, Dict, Any

from LLM import LLMInterface
from LLM.llm_constants import LLMProviderName, LLMAPIError, logger, DEFAULT_SYSTEM_MESSAGE, \
    ROLE_USER
from config import Config


class AnthropicLLM(LLMInterface):
    def __init__(self):
        super().__init__(LLMProviderName.ANTHROPIC)

    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        model = self.get_config_value(kwargs, "provider", Config)
        system_message_content = kwargs.get('system_message', DEFAULT_SYSTEM_MESSAGE)

        prompt_content = f"{context}\n\nSoru: {question}" if context else question

        payload: Dict[str, Any] = {
            "provider": model,
            "messages": [{"role": ROLE_USER, "content": prompt_content}],
        }
        if system_message_content:
            payload["system"] = system_message_content  # Anthropic 'system' parametresini ayrı alır

        common_params = self._prepare_common_payload_params(**kwargs)  # temperature, max_tokens
        payload.update(common_params)
        # Anthropic'e özel olabilecek diğer parametreler
        payload.update({k: v for k, v in kwargs.items() if k in ['top_p', 'top_k'] and v is not None})
        return payload

    def _parse_response(self, response_data: Any) -> str:
        # Claude 3 yanıt formatı: result['content'][0]['text']
        if response_data.get("content") and isinstance(response_data["content"], list) and len(
                response_data["content"]) > 0:
            if response_data["content"][0].get("type") == "text":
                return response_data["content"][0]["text"]
        # Eski format veya hata durumu
        logger.warning(f"{self.provider_name.capitalize()} API'den beklenmedik yanıt formatı: {response_data}")
        raise LLMAPIError(message="Yanıt formatı ayrıştırılamadı.", provider=self.provider_name,
                          response_text=str(response_data))
