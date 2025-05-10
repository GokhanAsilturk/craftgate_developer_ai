from typing import Optional, Dict, Any

from LLM.llmInterface import LLMInterface
from LLM.llm_constants import LLMProviderName, DEFAULT_SYSTEM_MESSAGE, ROLE_SYSTEM, ROLE_USER
from config import Config


class OpenAILLM(LLMInterface):
    def __init__(self):
        super().__init__(LLMProviderName.OPENAI)

    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        model = self.get_config_value(kwargs, "model", Config)
        system_message_content = kwargs.get('system_message', DEFAULT_SYSTEM_MESSAGE)

        messages = [{"role": ROLE_SYSTEM, "content": system_message_content}]
        if context:
            # OpenAI için context'i kullanıcı mesajının bir parçası olarak vermek daha yaygın
            full_question = f"Bağlam:\n{context}\n\nSoru: {question}"
            messages.append({"role": ROLE_USER, "content": full_question})
        else:
            messages.append({"role": ROLE_USER, "content": question})

        payload = {
            "provider": model,
            "messages": messages,
        }
        # Ortak ve modele özel parametreleri ekle
        common_params = self._prepare_common_payload_params(**kwargs)
        payload.update(common_params)
        # OpenAI'ye özel olabilecek diğer parametreler
        payload.update({k: v for k, v in kwargs.items() if
                        k in ['top_p', 'frequency_penalty', 'presence_penalty', 'seed'] and v is not None})
        return payload

    def _parse_response(self, response_data: Any) -> str:
        return response_data['choices'][0]['message']['content']
