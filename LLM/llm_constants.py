# LLM/llm_constants.py
import enum
import logging

# Loglama
logger = logging.getLogger(__name__)

# Sabitler
CONTENT_TYPE_JSON = "application/json"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
DEFAULT_SYSTEM_MESSAGE = ("Sen yardımcı bir asistansın. "
                          "Craftgate geliştirici dokümantasyonu hakkında sorulara yanıt ver."
                          "'Bu dökümanda, Bu belgede' gibi ifadeler kullanmadan yanıt ver."
                          "Sana verdiğimiz html sayfalarını kullanarak yanıt vermeye çalış."
                          "Soruyla ilgili yeterli bilgi yoksa yanıtın başına 'FALSE -' ekle."
                          "Bilgi yeterliyse cevabını 'TRUE -' ile başlat.")


# Enum sınıfı
class LLMProviderName(enum.Enum):
    """LLM sağlayıcı adları için enum sınıfı."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


# Hata sınıfları
class LLMConfigurationError(Exception):
    """LLM yapılandırması ile ilgili hatalar için özel exception sınıfı."""
    pass


class LLMAPIError(Exception):
    """LLM API çağrıları sırasında oluşan hatalar için özel exception sınıfı."""

    def __init__(self, provider: str, message: str, status_code: int = None, response_text: str = None):
        """
        Args:
            provider: Hata oluşan sağlayıcının adı
            message: Hata mesajı
            status_code: Hata durum kodu (opsiyonel)
            response_text: API'dan dönen ham yanıt (opsiyonel)
        """
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)

    def __str__(self):
        """
        Hata mesajının string gösterimi.
        Mesajı, durum kodunu ve ham yanıtı içerir.
        """
        error_parts = [f"LLM API Hatası ({self.provider}): {self.message}"]

        if self.status_code is not None:
            error_parts.append(f"Durum Kodu: {self.status_code}")

        if self.response_text:
            # Çok uzun yanıtları kısaltma
            max_response_length = 200
            response_preview = (self.response_text[:max_response_length] + '...'
                                if len(self.response_text) > max_response_length
                                else self.response_text)
            error_parts.append(f"Yanıt: {response_preview}")

        return "\n".join(error_parts)
