import requests
from typing import List
from config import Config

class LLMService:
    @staticmethod
    def generate_answer(question: str, context_sentences: List[str]) -> str:
        if not context_sentences:
            return "Bunu bulamadım."

        context = " ".join(context_sentences)
        prompt = f"""Soru: {question}
Bağlam: {context}

İşte talimatlar:
1. 'Bağlam' bölümündeki bilgilere dayanarak 'Soru'yu doğrudan yanıtla.
2. 'Verdiğiniz metinde', 'buradaki bilgiye göre' veya benzer ifadeler kullanma.
3. Doğrudan, net ve bilgiye dayalı bir cevap ver.
4. Bağlamda bulunan gerçek bilgilere sadık kal.
5. Eğer cevabı bulamıyorsan, sadece 'Bunu bulamadım.' yaz.

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