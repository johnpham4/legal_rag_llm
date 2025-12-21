import re

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_legal_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace('\xa0', ' ').replace('\r', '')

    # Remove excessive dashes and asterisks
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'\*{3,}', '', text)

    # Remove page numbers and footer patterns
    text = re.sub(r'Trang\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)

    # Remove signature blocks and administrative sections (NOT legal content)
    # These patterns appear at the end of documents
    removal_patterns = [
        r'Nơi nhận:.*?(?=\n\n|\Z)',  # "Nơi nhận:" section
        r'KT\..*?THỦ TRƯỞNG.*?(?=\n\n|\Z)',  # "KT. BỘ TRƯỞNG\nTHỦ TRƯỞNG"
        r'TM\..*?(?=\n[A-ZĐẤƯỨ]|\Z)',  # "TM. UỶ BAN NHÂN DÂN"
        r'(?:CHỦ TỊCH|GIÁM ĐỐC|TRƯỞNG BAN)\s*\n\s*[A-ZĐẮẰẲẴẶĂẤẦẨẪẬÂÁÀẢÃẠÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zđắằẳẵặăấầẩẫậâáàảãạéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]+(?=\n\n|\Z)',  # Signature name after title
    ]

    for pattern in removal_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Normalize legal structure markers
    # Điều X: or Điều X. -> Điều X: (SKIP if preceded by "như" or "Như")
    text = re.sub(r'(?<![tT]ại\s)Điều\s+(\d+)[\.\s]*', r'\nĐiều \1: ', text)

    # Normalize Mục (Roman numerals) - ensure newline before them
    text = re.sub(r'[\n\s]+(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.\s+', r'\n\1. ', text)

    # Khoản patterns: "1.", "2.", etc (NOT after "Điều")
    text = re.sub(r'(?<!Điều\s)(?<!Điều\s\s)[\n\s]+(\d+)\.\s+', r'\nKhoản \1. ', text)

    # Điểm patterns: "a)", "b)", "c)" etc
    text = re.sub(r'\n([a-z])\)\s+', r'\nĐiểm \1) ', text)

    # Normalize chapter markers
    text = re.sub(r'Chương\s+([IVXLCDM]+|[\d]+)[\.\:\s]*', r'\nChương \1: ', text, flags=re.IGNORECASE)

    # Clean multiple spaces but preserve single line breaks
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    text = '\n'.join(line for line in cleaned_lines if line)

    # Remove excessive line breaks (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()