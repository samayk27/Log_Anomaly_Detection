"""
Log Preprocessing Pipeline: lowercase, remove timestamps/special chars,
tokenization, stopword removal, message cleaning.
"""
import re
from typing import List

# Common log stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'the',
    'info', 'debug', 'success', 'request', 'response',
    'http', 'https', 'com', 'org', 'net', 'ms', 'us', 'ok',
    'true', 'false', 'null', 'undefined',
}

# Timestamp patterns to remove
TIMESTAMP_PATTERNS = [
    re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'),
    re.compile(r'\d{2}:\d{2}:\d{2}(?:\.\d+)?'),
    re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}'),
    re.compile(r'\[.*?\]'),  # bracketed content like [INFO]
]

# Special chars to remove or replace
SPECIAL_CHARS = re.compile(r'[^\w\s]')
EXTRA_SPACES = re.compile(r'\s+')


def lowercase_normalize(text: str) -> str:
    """Convert to lowercase."""
    return text.lower().strip()


def remove_timestamps(text: str) -> str:
    """Remove timestamp-like patterns."""
    result = text
    for pat in TIMESTAMP_PATTERNS:
        result = pat.sub(' ', result)
    return result


def remove_special_chars(text: str, keep_spaces: bool = True) -> str:
    """Remove or replace special characters."""
    if keep_spaces:
        return SPECIAL_CHARS.sub(' ', text)
    return SPECIAL_CHARS.sub('', text)


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return EXTRA_SPACES.sub(' ', text).strip().split()


def remove_stopwords(tokens: List[str], custom_stopwords: set | None = None) -> List[str]:
    """Remove stopwords from token list."""
    stop = STOPWORDS | (custom_stopwords or set())
    return [t for t in tokens if t.lower() not in stop]


def clean_message(raw: str, remove_stop: bool = True) -> str:
    """
    Full preprocessing pipeline:
    1. lowercase
    2. remove timestamps and bracketed content
    3. remove special characters
    4. tokenize
    5. remove stopwords
    6. rejoin to cleaned message
    """
    if not raw or not isinstance(raw, str):
        return ''
    t = lowercase_normalize(raw)
    t = remove_timestamps(t)
    t = remove_special_chars(t)
    tokens = tokenize(t)
    if remove_stop:
        tokens = remove_stopwords(tokens)
    return ' '.join(tokens) if tokens else ''


def process_logs(log_entries: List[dict]) -> List[dict]:
    """
    Process a list of parsed log dicts, adding 'cleaned_message' field.
    """
    result = []
    for entry in log_entries:
        msg = entry.get('message') or entry.get('raw', '')
        level = entry.get('log_level', '')
        entry_copy = dict(entry)
        entry_copy['cleaned_message'] = clean_message(f'{level} {msg}')
        result.append(entry_copy)
    return result


if __name__ == '__main__':
    samples = [
        "2026-02-17 08:00:10 INFO Block 1023 replicated to 3 nodes successfully",
        "ERROR Connection timeout after 12000ms to host 10.0.12.45",
        '{"timestamp":"2026-01-01T09:00:00","level":"INFO","message":"Email queue processed"}',
    ]
    for s in samples:
        print(repr(s[:60]), '->', clean_message(s))
