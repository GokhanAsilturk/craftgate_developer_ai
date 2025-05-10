# crawler.py
import json
import logging
import os
import re
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import Config

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


CACHE_FILE = "craftgate_crawled_contents.json"
CACHE_TTL = 24 * 3600  # 24 saat


def load_cached() -> Optional[List[Dict[str, str]]]:
    if not os.path.exists(CACHE_FILE):
        return None
    if time.time() - os.path.getmtime(CACHE_FILE) > CACHE_TTL:
        return None
    with open(CACHE_FILE, encoding="utf-8") as f:
        logger.info("🗄️ Cache yüklendi, tarama atlandı.")
        return json.load(f)


def save_cache(chunks: List[Dict[str, str]]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("🗄️ Yeni tarama tamamlandı, cache'e kaydedildi.")


def crawl_website_with_cache(base_url: str, force_refresh: bool = False) -> List[Dict[str, str]]:
    if not force_refresh:
        cached = load_cached()
        if cached is not None:
            return cached

    # force_refresh=True ya da cache yaşlıysa buraya düşer
    chunks = crawl_website(base_url)
    save_cache(chunks)

    # Debug için içerikleri bir txt dosyasına kaydedelim
    # with open("crawled_contents.txt", "w", encoding="utf-8") as f:
    #     for i, chunk in enumerate(chunks):
    #         f.write(f"URL: {chunk['url']}\n")
    #         f.write(f"Başlık: {chunk.get('title', '')}\n")
    #         f.write(f"İçerik: {chunk['text'][:500]}...\n\n")

    return chunks


def crawl_website(base_url: str) -> List[Dict[str, str]]:
    """
    Web sitesi sayfalarını tarar ve içerik+HTML şeklinde döndürür.
    """
    visited = set()
    to_visit = [base_url]
    all_pages = []
    seen_urls = set()

    while to_visit:
        url = to_visit.pop(0)
        normalized_url = normalize_url(url)

        if normalized_url in visited or normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)

        if '/en' in urlparse(url).path:
            continue

        try:
            logger.info(f"Sayfa taranıyor: {url}")
            response = requests.get(url, headers={"User-Agent": Config.USER_AGENT}, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue

            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            # Dil kontrolü yap
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang') != 'tr':
                continue

            # İçeriği çıkart
            title = soup.title.text.strip() if soup.title else ""

            # Ana içeriği çıkar
            main_content = extract_main_content(html_content)

            # HTML'i temizle
            simplified_html = simplify_html(html_content)

            # Sayfayı sakla
            page = {
                'url': url,
                'title': title,
                'text': main_content,  # Vektör aramalar için metin içerik
                'html': simplified_html,  # LLM için HTML içerik
                'content_index': len(all_pages)  # İndeks bilgisi
            }

            all_pages.append(page)
            print(f"Çekilen içerik: {title[:50]}... (URL: {url})")

            # Linkleri ekle
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)
                parsed_link = urlparse(absolute_url)
                if parsed_link.netloc == urlparse(base_url).netloc and '/en' not in parsed_link.path:
                    to_visit.append(absolute_url)

            visited.add(normalized_url)

        except Exception as e:
            logger.error(f"Tarama hatası {url}: {e}")

    logger.info(f"Toplam {len(all_pages)} sayfa çekildi.")
    return all_pages


def extract_main_content(html_content: str):
    """
    HTML içeriğinden ana metni metin olarak çıkarır.
    """
    # Gelen HTML string'inden BeautifulSoup objesi oluştur
    soup = BeautifulSoup(html_content, 'html.parser')

    # Ana içerik containerları
    main_selectors = [
        'main', 'article', 'div.content', 'div.main-content',
        'div#content', 'section.content', 'div[role="main"]'
    ]

    main_content_text = ""  # Ana metni tutacak değişken

    for selector in main_selectors:
        main_content_element = soup.select_one(selector)
        if main_content_element:
            # Sadece get_text() çağırın, sonra temizleyin
            main_content_text = main_content_element.get_text()
            break  # İlk bulunan ana içeriği alıp döngüyü kır

    # Eğer ana içerik selectorları ile bir şey bulunamazsa fallback'e geç
    if not main_content_text:
        content_divs = [(div, len(div.get_text())) for div in soup.find_all('div')
                        if len(div.get_text().strip()) > 100]

        if content_divs:
            # İçerik en uzun olan div'i bul
            max_div = max(content_divs, key=lambda x: x[1])
            # Sadece get_text() çağırın, sonra temizleyin
            main_content_text = max_div[0].get_text()
        else:
            # Son çare: Tüm metni al (simplify_html zaten temizlenmiş HTML verdiğini varsayıyoruz)
            # Sadece get_text() çağırın, sonra temizleyin
            main_content_text = soup.get_text()

    # Metni temizle: Fazla boşlukları ve satır sonlarını tek boşluğa dönüştür ve baştaki/sondaki boşlukları kırp
    cleaned_text = re.sub(r'\s+', ' ', main_content_text).strip()

    return cleaned_text

def simplify_html(html_content):
    """
    HTML'i sadeleştir, sadece <article> içeriğini (veya fallback olarak <main>/<body>) al,
    ve gereksiz kısımları kaldır.
"""
    soup = BeautifulSoup(html_content, 'html.parser')
    target_element = None

    # Önce <article> etiketini ara
    article_element = soup.find('article')
    if article_element:
        target_element = article_element
    else:
        # <article> yoksa <main> etiketini ara
        main_element = soup.find('main')
        if main_element:
            target_element = main_element
        else:
            # <main> de yoksa <body> etiketini kullan (varsa)
            target_element = soup.body

    # Eğer hedef element bulunduysa (article, main veya body)
    if target_element:
        # Hedef element içindeki gereksiz etiketleri kaldır
        for tag in target_element.select('script, style, meta, link, iframe, img, nav, footer'):
            # header ve footer gibi genel sarmalayıcıları da kaldıralım
            tag.decompose()

        # Temizlenmiş hedef elementin string temsilini döndür
        return str(target_element)
    else:
        # Hiçbir ana içerik elementi bulunamazsa (body dahil), tüm soup'un metnini döndür
        # veya boş bir string döndür - duruma göre karar verilebilir.
        # Şimdilik temizlenmiş tüm soup'u döndürelim (script/style vb. kaldırılmış haliyle)
        for tag in soup.select('script, style, meta, link, iframe, img, nav, footer'):
            tag.decompose()
        return str(soup)
