# vector_store.py
import logging
import re

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Metni normalleştir: fazla boşlukları kaldır, büyük-küçük harf farkını ortadan kaldır."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


class VectorStore:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.seen_texts = set()

    def clear(self):
        """Koleksiyonu sil ve yeniden oluştur."""
        try:
            self.client.delete_collection("documents")
            logger.info("Koleksiyon başarıyla silindi.")
        except Exception as e:
            logger.error(f"Koleksiyon silinirken hata oluştu: {e}")
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.seen_texts = set()
        logger.info("Yeni koleksiyon oluşturuldu.")

    def add_chunks(self, chunks):
        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for chunk in chunks:
            text = normalize_text(chunk['text'])
            if text in self.seen_texts:
                continue
            self.seen_texts.add(text)

            embedding = self.encoder.encode(text, show_progress_bar=True)
            documents.append(text)
            metadatas.append(chunk)
            embeddings.append(embedding)
            ids.append(f"{chunk['url']}_{chunk['content_index']}")

        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

    def query_similar(self, query, top_k=10):
        query = normalize_text(query)
        query_embedding = self.encoder.encode(query, show_progress_bar=True)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
