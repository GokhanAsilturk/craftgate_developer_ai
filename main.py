# main.py
import argparse
import logging
from config import Config
from content_processor import TextProcessor
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

    print("\nÇekilen içerikler 'crawled_contents.txt' dosyasına kaydedildi. Lütfen dosyayı kontrol edin.")
    vector_store = VectorStore()

    # Vektör veritabanını temizle
    vector_store.clear()

    vector_store.add_chunks(chunks)

    while True:
        query = input("Sorunuz (çıkmak için 'çık'): ")
        if query.lower() == "çık":
            break

        # Sorgudan anahtar kelimeleri çıkar
        keywords = TextProcessor.extract_keywords(query)
        logger.info(f"Çıkarılan anahtar kelimeler: {keywords}")
        # Anahtar kelimeleri birleştirerek yeni bir sorgu oluştur
        refined_query = " ".join(keywords)
        logger.info(f"İyileştirilmiş sorgu: {refined_query}")

        query_results = vector_store.query_similar(refined_query)
        distances = query_results['distances'][0]
        selected_chunks = query_results['metadatas'][0]

        # Cosine distance için 1'den çıkarıyoruz (1-mesafe = benzerlik)
        similarities = [1 - distance for distance in distances]
        logger.info(f"Benzerlik skorları: {similarities}")

        # Benzerlik skoru eşik değerinden büyük olanları seçiyoruz (benzerlik = 1-mesafe)
        relevant_chunks = [chunk for chunk, similarity in zip(selected_chunks, similarities)
                           if similarity >= 0.2]  # 0.2 düşük bir benzerlik eşiği

        if not relevant_chunks:
            print("Bunu bulamadım.")
            print("\nEn yakın içerik parçaları (eşleşme sağlanamadı):")
            for chunk, similarity in zip(selected_chunks[:3], similarities[:3]):
                print(f"- Paragraf: {chunk['text'][:50]}... (Benzerlik: {similarity:.4f})")
            continue

        context_sentences = [chunk['text'] for chunk in relevant_chunks]
        answer = LLMService.generate_answer(query, context_sentences)
        print(f"Yanıt: {answer}")

        print("\nSeçilen paragrafların benzerlikleri:")
        relevant_similarities = [1 - distance for chunk, distance in zip(relevant_chunks, distances[:len(relevant_chunks)])]
        for chunk, similarity in zip(relevant_chunks, relevant_similarities):
            print(f"- Paragraf: {chunk['text'][:50]}... (Benzerlik: {similarity:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true",
                        help="Mevcut cache varsa bile taramayı yeniden yap")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
