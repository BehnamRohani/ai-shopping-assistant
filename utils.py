import re

# Persian/Arabic numerals → English
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"

PERSIAN_TO_EN = str.maketrans(PERSIAN_DIGITS + ARABIC_DIGITS, ENGLISH_DIGITS * 2)

# Common Persian characters to unify
NORMALIZATION_MAP = {
    "ي": "ی",   # Arabic Yeh → Persian Yeh
    "ك": "ک",   # Arabic Kaf → Persian Kaf
    "ۀ": "ه",   # Heh with small Yeh → Heh
    "ة": "ه",   # Arabic Teh Marbuta → Heh
    "ؤ": "و",   # Waw with Hamza → Waw
    "إ": "ا",   # Alef with Hamza below → Alef
    "أ": "ا",   # Alef with Hamza above → Alef
    "آ": "ا",   # Alef Madda → Alef (optional)
}

def preprocess_persian(text: str) -> str:
    """
    Normalize Persian text:
    - Convert Persian/Arabic digits to English
    - Normalize Arabic variants of Persian letters
    - Remove extra whitespace
    - Standardize punctuation
    """
    # 1. Convert digits
    text = text.translate(PERSIAN_TO_EN)
    
    # 2. Normalize letters
    for src, dst in NORMALIZATION_MAP.items():
        text = text.replace(src, dst)
    
    # 3. Standardize punctuation
    text = text.replace("،", ",")  # Persian comma
    text = text.replace("؛", ";")  # Persian semicolon
    text = text.replace("؟", "?")  # Persian question mark
    
    # 4. Remove tatweel/kashida
    text = text.replace("ـ", "")
    
    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
