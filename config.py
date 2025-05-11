# config.py
class Config:
    BASE_URL = "https://developer.craftgate.io/"
    EMBEDDING_MODEL = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'

    # Varsayılan LLM sağlayıcı ayarı
    DEFAULT_LLM_PROVIDER = "gemini"  # Kullanılabilir değerler: "openai", "anthropic", "gemini", "ollama", "huggingface"

    # Her sağlayıcı için provider ayarları
    LLM_MODELS = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-sonnet",
        "gemini": "gemini-2.0-flash",
        "huggingface": "microsoft/phi-4-mini-instruct",
        "ollama": "mistral:7b"
    }

    # API anahtarları (güvenlik için çevresel değişkenlerden almanız daha iyi olur)
    API_KEYS = {
        "openai": "your_openai_api_key",
        "anthropic": "your_anthropic_api_key",
        "gemini": "AIzaSyBJQsad_rLYezFVWLckk1y-WqIYpd5xags",
        "huggingface": "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"
    }

    # API endpoint'leri
    API_URLS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "huggingface": "https://api-inference.huggingface.co/models/",
        "ollama": "http://localhost:11434/api/generate"
    }

    # Ortak LLM parametreleri
    LLM_PARAMS = {
        "temperature": 0.7,  # Yaratıcılık seviyesi (0.0-1.0)
        "max_tokens": 1000,  # Maksimum token sayısı
        "timeout": 300  # Zaman aşımı (saniye)
    }

    # Model özel parametreler (gerekirse)
    MODEL_SPECIFIC_PARAMS = {
        "ollama": {
            "stream": False  # Ollama'ya özel parametre
        }
    }

    # Diğer ayarlar
    COLLECTION_NAME = "site_chunks"
    RELEVANCE_THRESHOLD = 0.6
    MAX_RESULTS = 10
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    STOPWORDS = {'nedir', 'nasıl', 'ne', 'için', 'ile', 've', 'bir', 'bu', 'şu', 'zaman'}

    # Benzerlik eşiği ayarı
    MIN_SIMILARITY_THRESHOLD = 0.5

    # Qdrant ayarları
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    QDRANT_COLLECTION = "craftgate_docs"
