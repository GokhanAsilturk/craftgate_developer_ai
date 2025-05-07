# llm_service.py
from typing import Dict

import requests

from config import Config


class LLMService:
    @staticmethod
    def generate_answer_from_html(question: str, page_data: Dict) -> str:
        """HTML içeriğinden yanıt üretir."""
        if not page_data or 'html' not in page_data:
            return "Bunu bulamadım."

        html_content = page_data['html']
        url = page_data['url']
        
        prompt = f"""Soru: {question}
URL: {url}
HTML İçeriği:
{html_content}

İşte talimatlar:
1. Yukarıdaki HTML içeriğinden soruyu doğrudan yanıtla.
2. HTML yapısını kullanarak başlıkları, listeleri ve diğer öğeleri doğru bağlamda anla.
3. 'Verdiğiniz metinde', 'buradaki bilgiye göre' veya benzer ifadeler kullanma.
4. Doğrudan, net ve bilgiye dayalı bir cevap ver.
5. HTML içeriğindeki gerçek bilgilere sadık kal.
6. Eğer cevabı bulamıyorsan, sadece 'Bunu bulamadım.' yaz.

Yanıt:"""

        try:
            response = requests.post(
                Config.OLLAMA_API_URL,
                json={"model": Config.LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=Config.OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("response", "Bunu bulamadım.").strip()
        except Exception as e:
            print(f"LLM hata: {e}")
            return "Bunu bulamadım."