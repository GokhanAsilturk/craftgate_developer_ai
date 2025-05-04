# crawler.py
import json
import os
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import logging
from config import Config
import re

logger = logging.getLogger(__name__)

def normalize_url(url: str) -> str:
    """URL'yi normalleştir: query ve fragmentleri kaldır."""
    parsed = urlparse(url)
    return parsed.scheme + "://" + parsed.netloc + parsed.path

def normalize_text(text: str) -> str:
    """Metni normalleştir: fazla boşlukları kaldır, büyük-küçük harf farkını ortadan kaldır."""
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları kaldır
    text = text.lower()  # Büyük-küçük harf farkını kaldır
    return text

CACHE_FILE = "crawled_contents.json"
CACHE_TTL = 24 * 3600  # 24 saat

def load_cached() -> Optional[List[Dict[str,str]]]:
    if not os.path.exists(CACHE_FILE):
        return None
    if time.time() - os.path.getmtime(CACHE_FILE) > CACHE_TTL:
        return None
    with open(CACHE_FILE, encoding="utf-8") as f:
        logger.info("🗄️ Cache yüklendi, tarama atlandı.")
        return json.load(f)

def save_cache(chunks: List[Dict[str,str]]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("🗄️ Yeni tarama tamamlandı, cache'e kaydedildi.")

def crawl_website_with_cache(base_url: str, force_refresh: bool=False) -> List[Dict[str,str]]:
    if not force_refresh:
        cached = load_cached()
        if cached is not None:
            return cached

    # force_refresh=True ya da cache yaşlıysa buraya düşer
    chunks = crawl_website(base_url)
    save_cache(chunks)
    return chunks

def crawl_website(base_url: str) -> List[Dict[str, str]]:
    visited = set()
    to_visit = [base_url]
    all_chunks = []
    seen_texts = set()

    while to_visit:
        url = to_visit.pop(0)
        normalized_url = normalize_url(url)
        if normalized_url in visited:
            continue

        if '/en' in urlparse(url).path:
            continue

        try:
            response = requests.get(url, headers={"User-Agent": Config.USER_AGENT})
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang') != 'tr':
                continue

            content_sections = []
            current_section = []
            current_header = None

            for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'pre', 'code', 'li', 'td', 'div', 'span']):
                text = tag.get_text().strip()
                if not text:
                    continue

                if tag.name in ['h1', 'h2', 'h3']:
                    if current_section and current_header:
                        content_sections.append({
                            'header': current_header,
                            'content': " ".join(current_section)
                        })
                    current_header = text
                    current_section = []
                else:
                    current_section.append(text)

            if current_section and current_header:
                content_sections.append({
                    'header': current_header,
                    'content': " ".join(current_section)
                })

            for idx, section in enumerate(content_sections):
                # Başlık ve içeriği normalleştir
                header = normalize_text(section['header'])
                content = normalize_text(section['content'])

                # Başlık içeriğin içinde tekrar ediyorsa, bunu temizle
                if header in content:
                    content = content.replace(header, '').strip()

                chunk_text = f"{section['header']}: {content}"
                if chunk_text in seen_texts:
                    continue
                seen_texts.add(chunk_text)

                # "ödeme", "form", "başlatma" kelimelerini içeren başlıklar ve içerikler
                header_lower = header
                content_lower = content
                if True:  # Tüm içeriği al veya daha kapsamlı filtre uygula
                    print(f"Çekilen içerik: {chunk_text[:100]}... (URL: {url})")
                    all_chunks.append({
                        'url': url,
                        'content_index': idx,
                        'text': chunk_text
                    })

            visited.add(normalized_url)

            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)
                parsed_link = urlparse(absolute_url)
                if parsed_link.netloc == urlparse(base_url).netloc and '/en' not in parsed_link.path:
                    to_visit.append(absolute_url)

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")

    with open("crawled_contents.txt", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(f"URL: {chunk['url']}\n")
            f.write(f"İçerik: {chunk['text'][:500]}...\n\n")

    logger.info(f"Toplam {len(all_chunks)} içerik parçası çekildi.")
    return all_chunks