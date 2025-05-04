# content_processor.py
from typing import List
from config import Config

class TextProcessor:
    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in text.split('\n') if p.strip()]

    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        words = query.lower().split()
        return [word for word in words if word not in Config.STOPWORDS]