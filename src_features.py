import re
from urllib.parse import urlparse

from sklearn.base import BaseEstimator, TransformerMixin

URL_PATTERN = re.compile(r"https?://\\S+|www\\.\\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", re.IGNORECASE)

SUSPICIOUS_WORDS = {
    "urgent", "verify", "verification", "suspended", "suspend", "password",
    "winner", "prize", "claim", "refund", "billing", "payment", "confirm",
    "immediately", "login", "security", "expired", "expire", "action required",
    "bank", "credential", "limited", "blocked", "update"
}


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract simple indicators from email text; these are not proof of phishing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._features(str(text)) for text in X]

    @staticmethod
    def _features(text):
        lower = text.lower()
        urls = URL_PATTERN.findall(text)
        domains = []
        for raw_url in urls:
            candidate = raw_url.rstrip(".,);!?]")
            if candidate.lower().startswith("www."):
                candidate = "http://" + candidate
            try:
                domains.append(urlparse(candidate).hostname or "")
            except ValueError:
                domains.append("")

        words = re.findall(r"\\b[\\w']+\\b", lower)
        suspicious_count = sum(1 for word in words if word in SUSPICIOUS_WORDS)
        return {
            "char_count": len(text),
            "word_count": len(words),
            "url_count": len(urls),
            "email_address_count": len(EMAIL_PATTERN.findall(text)),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "uppercase_ratio": sum(ch.isupper() for ch in text) / max(1, sum(ch.isalpha() for ch in text)),
            "digit_count": sum(ch.isdigit() for ch in text),
            "suspicious_keyword_count": suspicious_count,
            "has_http_url": int(any(u.lower().startswith("http://") for u in urls)),
            "has_ip_like_url": int(bool(re.search(r"https?://(?:\\d{1,3}\\.){3}\\d{1,3}", text))),
            "has_at_symbol": int("@" in text),
            "has_shortened_url_hint": int(any(
                any(short in domain.lower() for short in ("bit.ly", "tinyurl", "t.co"))
                for domain in domains
            )),
        }
