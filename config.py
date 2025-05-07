# config.py
class Config:
    BASE_URL = "https://developer.craftgate.io"
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    LLM_MODEL = "gemma3:4b"
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_TIMEOUT = 300
    COLLECTION_NAME = "site_chunks"
    RELEVANCE_THRESHOLD = 0.6
    MAX_RESULTS = 10
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    STOPWORDS = {'nedir', 'nasıl', 'ne', 'için', 'ile', 've', 'bir', 'bu', 'şu', 'zaman', '?'}