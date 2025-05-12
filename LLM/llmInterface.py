# LLM/llmInterface.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any  # Type hintler için Type eklendi

import requests  # requests kütüphanesi eklendi

# Constants ve hatalar için importlar (bunlar muhtemelen zaten var)
from LLM.llm_constants import logger, LLMAPIError, LLMConfigurationError, CONTENT_TYPE_JSON, \
    LLMProviderName  # LLMProviderName de eklendi


# Config objesine erişim için, get_config_value metodu Config'i parametre olarak alıyor, bu iyi.
# Ancak generate_answer metodu içinde kullanmak için Config'i import edelim (metod içinde import dairesel riski azaltır)
# from config import Config # Generate answer metodu içinde import edilecek

class LLMInterface(ABC):
    """Tüm LLM sağlayıcıları için temel sınıf."""

    def __init__(self, provider_name: LLMProviderName):  # LLMProviderName tipini kullanabiliriz
        """
        Args:
            provider_name: Sağlayıcı enum değeri (LLMProviderName)
        """
        # __init__ metodunda enum değerini string'e çevirerek saklayalım
        self.provider_name: str = provider_name.value

    def get_config_value(self, kwargs: dict, key: str, config_obj, default_value=None):
        """Yapılandırma değerini kwargs'dan veya Config'den alır."""
        # Önce kwargs'a bak
        if key in kwargs:
            return kwargs[key]

        # Sonra config nesnesine bak (Config sınıfı)
        # config_obj = Config olmalı generate_answer içinde
        if hasattr(config_obj, key.upper()):
            return getattr(config_obj, key.upper())

        # API/Model/Parametre sözlüklerine bak (Config.API_URLS, Config.LLM_MODELS, Config.LLM_PARAMS)
        # self.provider_name artık string olduğu için .value kullanmaya gerek yok
        if key == "api_url":
            return config_obj.API_URLS.get(self.provider_name)
        elif key == "api_key":
            return config_obj.API_KEYS.get(self.provider_name)
        elif key == "model":
            return config_obj.LLM_MODELS.get(self.provider_name)
        elif key in ["temperature", "max_tokens", "timeout"]:
            return config_obj.LLM_PARAMS.get(key)
        # Modele özel parametrelere bak
        elif hasattr(config_obj,
                     'MODEL_SPECIFIC_PARAMS') and self.provider_name in config_obj.MODEL_SPECIFIC_PARAMS and key in \
                config_obj.MODEL_SPECIFIC_PARAMS[self.provider_name]:
            return config_obj.MODEL_SPECIFIC_PARAMS[self.provider_name].get(key)

        return default_value

    # Ortak HTTP isteği yapma metodu (stream destekli değil şimdilik, Ollama override edecek)
    # Bu metodun dönüş tipi requests.Response
    def _make_request(self, method: str, url: str, headers: Dict[str, str],
                      json_payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None,
                      **kwargs) -> requests.Response:  # requests.Response dönüş tipi eklendi
        """
        Merkezi HTTP istek fonksiyonu.

        Args:
            method: HTTP metodu (GET, POST)
            url: İstek URL'si
            headers: İstek başlıkları
            json_payload: POST/PUT istekleri için JSON gövdesi
            params: URL query parametreleri
            kwargs: Timeout gibi ek parametreler (generate_answer'dan gelir)

        Returns:
            requests.Response: İstek yanıt objesi

        Raises:
            LLMAPIError: API bağlantı veya HTTP hatası durumunda
        """
        # Config objesine erişim için, fonksiyon/metod içinde import etmek dairesel import riskini azaltır
        from config import Config  # Fonksiyon/metod içinde import etmek dairesel import riskini azaltır

        timeout = self.get_config_value(kwargs, "timeout", Config)

        try:
            response = requests.request(method, url, headers=headers, json=json_payload, params=params, timeout=timeout)
            # HTTP hatalarını (4xx, 5xx) otomatik olarak kontrol et
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            # HTTP hatası durumunda API'dan dönen yanıtı yakalamaya çalış
            response_text = None
            try:
                response_text = e.response.json()  # JSON parse etmeyi dene
            except:
                try:
                    response_text = e.response.text  # JSON değilse metin olarak al
                except:
                    pass  # Yanıt gövdesi yoksa

            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else None

            logger.error(
                f"{self.provider_name.capitalize()} API HTTP Hatası: {status_code} - {response_text}",
                exc_info=True)
            # LLMAPIError fırlatırken response_text'i string yapalım
            raise LLMAPIError(message="API çağrısı başarısız oldu.", provider=self.provider_name,
                              status_code=status_code, response_text=str(response_text)) from e
        except requests.exceptions.RequestException as e:
            # Bağlantı zaman aşımı gibi diğer requests hatalarını yakala
            logger.error(f"{self.provider_name.capitalize()} API Bağlantı Hatası: {e}", exc_info=True)
            raise LLMAPIError(message=f"API'ye bağlanırken bir sorun oluştu: {e}", provider=self.provider_name) from e


    @abstractmethod
    def _prepare_payload(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        API isteği için modele özel payload'ı hazırlar.
        Bu metod her sağlayıcı sınıfı tarafından implemente edilmelidir.
        """
        pass  # Implementasyon sağlayıcı sınıflarında olacak

    @abstractmethod
    def _parse_response(self, response_data: Any) -> str:
        """
        API yanıtından (JSON/dict olarak) üretilen metin yanıtını ayrıştırır.
        Bu metod her sağlayıcı sınıfı tarafından implemente edilmelidir.
        API'ye özel hataları (güvenlik filtresi vb.) burada LLMAPIError fırlatarak belirtebilirsiniz.
        """
        pass  # Implementasyon sağlayıcı sınıflarında olacak

    def generate_answer(self, question: str, context: str = None, **kwargs) -> str:
        """
        Verilen soru ve bağlam üzerinden ortak akışla yanıt üretir.
        Bu metod genellikle override edilmeyecektir (Hugging Face URL yapısı veya Ollama stream gibi özel durumlar hariç).
        """
        # Config objesine erişim için import (metod içinde)
        from config import Config

        try:
            # 1. Payload'ı hazırlayın (sağlayıcıya özel metod)
            # _prepare_payload metoduna tüm kwargs'ları iletelim ki içinde config değerlerini alabilsin
            payload = self._prepare_payload(question, context, **kwargs)

            # 2. API URL'sini ve API Key'i alın
            # _prepare_payload içinde URL hazırlama yoksa Config'den alınır
            api_url = self.get_config_value(kwargs, "api_url", Config)
            api_key = self.get_config_value(kwargs, "api_key", Config)
            # Model adı payload'da belirtildiği için burada ayrıca alınmasına gerek yok

            # API URL eksikse hata fırlat
            if not api_url:
                raise LLMConfigurationError(
                    f"{self.provider_name.capitalize()} için API URL'si Config.API_URLS'de tanımlı değil veya _prepare_payload içinde hazırlanmadı.")

            # 3. Headers'ı hazırlayın (API Key'i içeren)
            headers = {
                "Content-Type": CONTENT_TYPE_JSON,  # Genellikle JSON gönderilir
            }
            # API Key header'ı sağlayıcıya göre değişir
            if api_key:
                if self.provider_name == LLMProviderName.OPENAI.value:
                    headers["Authorization"] = f"Bearer {api_key}"
                elif self.provider_name == LLMProviderName.GEMINI.value:
                    # Gemini için header formatı "x-goog-api-key" veya Authorization olabilir
                    # Config'deki API_URL'e göre format değişebilir (Google AI vs Google Cloud)
                    # Config'de kullanılan URL'ye göre burada doğru header seçilmelidir.
                    # Google AI için "x-goog-api-key", Google Cloud için "Authorization: Bearer"
                    # Basitlik için şimdilik "x-goog-api-key" varsayalım (Config.API_URLS googleapis.com'a işaret ediyor)
                    headers["x-goog-api-key"] = api_key
                elif self.provider_name == LLMProviderName.ANTHROPIC.value:
                    headers["x-api-key"] = api_key
                    headers["anthropic-version"] = "2023-06-01"  # Anthropic version header'ı gerekebilir
                elif self.provider_name == LLMProviderName.HUGGINGFACE.value:
                    headers["Authorization"] = f"Bearer {api_key}"
                # Ollama genellikle API Key gerektirmez, gerektirirse buraya eklenir
                # else:
                #    # Bilinmeyen sağlayıcı için API Key header formatı nasıl olmalı?
                #    logger.warning(f"Sağlayıcı {self.provider_name} için API Key header formatı tanımsız.")

            # 4. HTTP isteği yapın (ortak metod)
            # _make_request metodu requests.exceptions hatalarını LLMAPIError'a sarmalar
            response = self._make_request("POST", api_url, headers=headers, json_payload=payload, **kwargs)

            # 5. Yanıt verisini alın (response.json() zaten _make_request içinde veya dışında yapılabilir)
            # _make_request response objesini döndürdüğü için burada .json() çağrısı yapmalıyız.
            response_data = response.json()

            # 6. Yanıtı ayrıştırın (sağlayıcıya özel metod)
            # _parse_response API'ye özel hataları (güvenlik vb.) LLMAPIError olarak fırlatmalı
            return self._parse_response(response_data)

        except LLMConfigurationError:
            # Config hatası zaten yakalandı ve loglandı/fırlatıldı, bunu tekrar fırlat
            raise
        except LLMAPIError:
            # _make_request veya _parse_response tarafından fırlatılan API hataları
            # Zaten LLMAPIError olduğu için bunu yakalayıp tekrar fırlatmak yeterli.
            # Daha spesifik bir işlem yapmak istenirse burası genişletilir.
            raise
        except Exception as e:
            # Beklenmeyen diğer hataları yakala
            logger.error(f"{self.provider_name.capitalize()} yanıt üretirken beklenmedik bir hata oluştu: {e}",
                         exc_info=True)
            # Beklenmeyen hataları da LLMAPIError'a sararak fırlat
            raise LLMAPIError(message=f"Beklenmedik bir hata oluştu: {e}", provider=self.provider_name) from e

    @staticmethod
    def get_error_message() -> str:
        """Hata durumunda döndürülecek genel mesaj."""
        return "Yanıt oluşturulurken bir hata meydana geldi."
