from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config


class VectorStore:
    def __init__(self):
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL, trust_remote_code=True)
        # Bellek içinde vektör depolama için veri yapıları
        self.vectors = []
        self.payloads = []
        self.initialized = True
        print("Hafıza tabanlı vektör deposu başlatıldı.")

    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            print("Eklenecek içerik parçası bulunamadı.")
            return

        # Metinleri vektörlere dönüştür
        embeddings = self.encoder.encode([chunk['text'] for chunk in chunks])

        # Vektörleri ve ilgili metadata'ları sakla
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.vectors.append(embedding)
            self.payloads.append({
                'url': chunk['url'],
                'text': chunk['text'],
                'html': chunk.get('html', '')
            })

        print(f"{len(chunks)} içerik parçası vektör deposuna eklendi.")

    def clear(self):
        # Eski verileri temizle
        self.vectors = []
        self.payloads = []
        print("Vektör deposu temizlendi.")

    def query_similar(self, query: str) -> Dict[str, List]:
        if not self.vectors:
            return {'metadatas': [], 'distances': []}

        # Sorgu vektörünü oluştur
        query_vector = self.encoder.encode(query)

        # Cosine benzerliklerini hesapla
        similarities = []
        for vec in self.vectors:
            # Cosine benzerliği hesapla: dot(a, b) / (||a|| * ||b||)
            similarity = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
            similarities.append(similarity)

        # En yüksek benzerliğe sahip olanları bul (sıralı indeksler)
        top_indices = np.argsort(similarities)[::-1][:Config.MAX_RESULTS]

        # Sonuçları hazırla
        selected_payloads = [self.payloads[i] for i in top_indices]
        # Uzaklık = 1 - benzerlik (cosine benzerliği için)
        selected_distances = [1 - similarities[i] for i in top_indices]

        return {
            'metadatas': selected_payloads,
            'distances': selected_distances
        }
