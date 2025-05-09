# llm_service.py
import logging
import re
from typing import Dict, List

from LLM.llm_providers import LLMFactory
from crawler import crawl_website_with_cache, extract_main_content

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

    @staticmethod
    def is_valid_answer(answer: str) -> bool:
        """Yanıtın geçerliliğini kontrol eder."""
        # Yanıtın uzunluğunu kontrol edin
        if len(answer) < 10:  # Çok kısa yanıtlar muhtemelen geçerli değildir
            return False

        # Yanıtın içeriğini kontrol edin
        invalid_patterns = [
            r"bilgi\s+bulunamadı",
            r"bilgim\s+yok",
            r"bunu\s+bulamadım",
            r"yanıt\s+(oluştur|üret).*?(hatası?|meydana\s+geldi)",
            r"geçerli\s+bir\s+yanıt\s+değil",
            r"bu\s+konuda\s+(bilgim|yeterli\s+bilgi)\s+(yok|bulunamadı)"
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return False

        return True

    @staticmethod
    def try_multiple_pages(question: str, page_urls: List[str], provider: str = None, model: str = None,
                           **kwargs) -> str:
        """
        Birden fazla sayfayı deneyerek en iyi yanıtı bulmaya çalışır.

        Args:
            question: Kullanıcı sorusu
            page_urls: Denenecek sayfaların URL'leri
            provider: Kullanılacak LLM sağlayıcısı
            model: Kullanılacak provider adı
            **kwargs: Diğer LLM parametreleri

        Returns:
            str: En iyi yanıt
        """
        if not page_urls:
            return "Aradığınız bilgiyi bulamadım."

        best_answer = ""

        for url in page_urls:
            try:
                # Sayfayı getirin
                page_data = crawl_website_with_cache(url)

                if not page_data or 'html' not in page_data:
                    continue

                # Yanıt oluşturun
                answer = LLMService.generate_answer_from_html(
                    question,
                    page_data,
                    provider=provider,
                    model=model,
                    **kwargs
                )

                # Yanıtın geçerliliğini kontrol edin
                if LLMService.is_valid_answer(answer):
                    best_answer = answer
                    break  # Geçerli bir yanıt bulundu, daha fazla aramaya gerek yok

                # Eğer daha önce hiç yanıt yoksa, geçerli olmasa bile bu yanıtı kaydedelim
                if not best_answer:
                    best_answer = answer

            except Exception as e:
                logger.error(f"URL {url} işlenirken hata oluştu: {str(e)}")
                continue

        return best_answer if best_answer else "Aradığınız bilgiyi bulamadım."
