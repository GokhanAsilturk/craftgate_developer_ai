# main.py
import argparse
import logging

from config import Config
from crawler import crawl_website_with_cache
from llm_service import LLMService
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

        similarities = [1 - distance for distance in distances]
        logger.info(f"Benzerlik skorları: {similarities}")

        # En iyi eşleşen sayfayı bul ve yanıt üret
        if similarities and similarities[0] >= 0.5:  # En az bir sonuç varsa ve benzerlik eşikten yüksekse
            best_match = selected_chunks[0]  # En iyi eşleşen sayfa (metadata)

            # HTML tabanlı yaklaşımı kullan
            answer = LLMService.generate_answer_from_html(query, best_match)
            print(f"Yanıt: {answer}")

            print(f"\nKullanılan sayfa: {best_match['url']}")
            print(f"Benzerlik skoru: %{similarities[0] * 100:.2f}")

        else:
            print("Bunu bulamadım.")
            if similarities:
                print("\nEn yakın içerik parçaları (eşleşme sağlanamadı):")
                # Eski davranış: Sadece benzerlikleri göster
                for chunk, similarity in zip(selected_chunks[:3], similarities[:3]):
                    # text anahtarının varlığını kontrol et
                    page_text = chunk.get('text', '[Metin Yok]')[:50]
                    page_url = chunk.get('url', '[URL Yok]')
                    print(f"- Sayfa: {page_url} (Benzerlik: {similarity * 100:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true",
                        help="Mevcut cache varsa bile taramayı yeniden yap")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
