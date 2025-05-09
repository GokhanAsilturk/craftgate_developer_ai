# main.py
import argparse
import logging

import numpy as np

from LLM.llm_service import LLMService
from config import Config
from crawler import crawl_website_with_cache
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(force_refresh: bool):
    logger.info("Web sitesi taranıyor (cache kontrol ediliyor)...")
    chunks = crawl_website_with_cache(Config.BASE_URL, force_refresh=force_refresh)
    if not chunks:
        print("Hiç içerik çekilemedi.")
        return

    vector_store = VectorStore()

    # Vektör veritabanını temizle
    vector_store.clear()

    vector_store.add_chunks(chunks)

    while True:
        query = input("Sorunuz (çıkmak için 'çık'): ")
        if query.lower() == "çık":
            break

        # Vektör araması için orijinal sorguyu kullan
        logger.info(f"Vektör araması için kullanılan sorgu: {query}")
        query_results = vector_store.query_similar(query)
        distances = query_results['distances'][0]
        selected_chunks = query_results['metadatas'][0]

        # distances'ın bir numpy dizisi veya tek bir değer olup olmadığını kontrol et
        if isinstance(distances, np.ndarray):
            similarities = [1 - distance for distance in distances]
        else:
            # Tek bir değer ise, onu bir listeye dönüştür
            similarities = [1 - distances]

        # Gelen `selected_chunks` değişkenini uygun formata dönüştür
        prepared_chunks = prepare_chunks(selected_chunks)

        logger.info(f"Benzerlik skorları: {similarities}")

        # Eğer yeterli sonuç varsa ve benzerlik eşiği karşılanıyorsa
        if similarities and similarities[0] >= Config.MIN_SIMILARITY_THRESHOLD:
            # Birden fazla sayfa deneyerek en iyi yanıtı bul
            max_pages_to_try = min(3, len(prepared_chunks))  # En fazla 3 sayfa deneyebiliriz

            answer, best_match, similarity, checked_pages = LLMService.try_multiple_pages(
                query,
                prepared_chunks[:max_pages_to_try],
                similarities[:max_pages_to_try]
            )

            print(f"Yanıt: {answer}")
            print("\nKontrol edilen sayfalar:")
            for page in checked_pages:
                print(f"- {page['url']} (Benzerlik: %{page['similarity'] * 100:.2f}) - {page['status']}")

        else:
            print("Bunu bulamadım.")
            if similarities:
                print("\nEn yakın içerik parçaları (eşleşme sağlanamadı):")

                # Liste dönüşümlerini yapalım
                chunks_list = list(selected_chunks) if hasattr(selected_chunks, '__iter__') else [selected_chunks]
                similarities_list = list(similarities) if hasattr(similarities, '__iter__') else [similarities]

                # Listelerden ilk 3 elemanı alalım (veya daha az varsa tamamını)
                chunks_list = chunks_list[:3] if len(chunks_list) >= 3 else chunks_list
                similarities_list = similarities_list[:3] if len(similarities_list) >= 3 else similarities_list

                # Şimdi güvenli bir şekilde zip işlemi yapabiliriz
                for chunk, similarity in zip(chunks_list, similarities_list):
                    # Chunk'ın string veya dict olma durumunu kontrol et
                    if isinstance(chunk, dict):
                        page_url = chunk.get('url', '[URL Yok]')
                    else:
                        # Eğer chunk bir string ise, doğrudan onu kullan
                        page_url = '[URL Bilgisi Yok]'

                    print(f"- Sayfa: {page_url} (Benzerlik: %{similarity * 100:.2f})")


def prepare_chunks(selected_chunks):
    """
    Gelen `selected_chunks` değişkenini işleyip bir `list` yapısına dönüştürür.
    Her bir öğe bir sözlük (`dict`) olmalıdır.
    """
    if isinstance(selected_chunks, list):
        return [
            chunk if isinstance(chunk, dict) else {'content': str(chunk), 'url': '[URL Bilgisi Yok]'}
            for chunk in selected_chunks
        ]
    elif isinstance(selected_chunks, dict):
        return [selected_chunks]  # Eğer dict ise listeye sararak döndür
    elif isinstance(selected_chunks, str):
        return [{'content': selected_chunks, 'url': '[URL Bilgisi Yok]'}]
    else:
        return [{'content': str(selected_chunks), 'url': '[URL Bilgisi Yok]'}]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true",
                        help="Mevcut cache varsa bile taramayı yeniden yap")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
