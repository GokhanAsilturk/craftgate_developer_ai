# llm_service.py
import logging
from typing import Dict

from LLM.llm_providers_config import LLMFactory
from crawler import extract_main_content

logger = logging.getLogger(__name__)


class LLMService:
    @staticmethod
    def generate_answer_from_html(question: str, page_data: Dict, provider: str = None, model: str = None,
                                  **kwargs) -> str:
        """
        HTML içeriğinden yanıt üretir.

        Args:
            question: Kullanıcı sorusu
            page_data: HTML içeriği ve diğer sayfa verileri
            provider: Kullanılacak LLM sağlayıcısı (ör. "openai", "anthropic", "gemini")
            model: Kullanılacak provider adı (ör. "gpt-4", "claude-instant")
            **kwargs: LLM sağlayıcısına özel parametreler
                      (temperature, api_key vb.)

        Returns:
            str: Üretilen yanıt
        """
        if not page_data or 'html' not in page_data:
            return "Bunu bulamadım."

        # HTML içeriğinden ana metni çıkarın
        html_content = page_data['html']
        main_content = extract_main_content(html_content)

        # Bağlam oluştur
        context = main_content

        # Parametreleri hazırla
        llm_kwargs = kwargs.copy()
        if model:
            llm_kwargs['provider'] = model

        # LLM sağlayıcısını oluştur
        llm = LLMFactory.create_llm(provider, **llm_kwargs)

        # Yanıt üret
        return llm.generate_answer(question, context, **llm_kwargs)





