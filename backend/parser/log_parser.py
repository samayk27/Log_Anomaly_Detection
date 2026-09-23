import re
import json
import csv
from io import StringIO
from typing import Any
from dataclasses import dataclass, asdict


@dataclass
class ParsedLog:
    timestamp: str
    service: str
    log_level: str
    message: str
    ip_address: str | None
    user_id: str | None
    status_code: str | None
    raw: str
    log_type: str
    dataset_label: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


APACHE_NGINX_RE = re.compile(
    r'^(\S+ \S+) (\S+) (\S+) \[([^\]]+)\] "([^"]*)" (\d{3}) (\d+|-)?'
)
JSON_TIMESTAMP_RE = re.compile(
    r'\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'
)
TIMESTAMP_RE = re.compile(
    r'^(?P<timestamp>(?:\d{4}[-/.]\d{2}[-/.]\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?'
    r'|\d{4}[-/.]\d{2}[-/.]\d{2}'
    r'|\d{2,6}[\s-]\d{2}:?\d{2}:?\d{2}'
    r'|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?))\s*'
)
ANY_TIMESTAMP_RE = re.compile(
    r'(?P<timestamp>\d{4}[-/.]\d{2}[-/.]\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?'
    r'|\d{4}[-/.]\d{2}[-/.]\d{2}'
    r'|\d{2,6}[\s-]\d{2}:?\d{2}:?\d{2}'
    r'|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)'
)
LEVEL_RE = re.compile(
    r'\b(EMERG|ALERT|CRIT|CRITICAL|FATAL|ERROR|ERR|WARNING|WARN|NOTICE|INFO|DEBUG|TRACE|SEVERE)\b', re.I
)
BGL_RE = re.compile(
    r'^\S+\s+\d+\s+\d{4}\.\d{2}\.\d{2}\s+\S+\s+'
    r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+\s+\S+\s+\S+\s+\S+\s+'
    r'(?:EMERG|ALERT|CRIT|CRITICAL|FATAL|ERROR|ERR|WARNING|WARN|NOTICE|INFO|DEBUG|TRACE|SEVERE)\b',
    re.I,
)
IP_RE = re.compile(
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
)
USER_ID_RE = re.compile(
    r'(?:user|userId|user_id|username)[\s=:]+[\'"]?([a-zA-Z0-9_-]+)[\'"]?',
    re.I
)
STATUS_CODE_RE = re.compile(
    r'(?:status|code|status_code)[\s=:]+(\d{3})', re.I
)
HTTP_STATUS_RE = re.compile(
    r'\b(200|201|301|302|400|401|403|404|500|502|503)\b'
)


def _normalize_level(raw: str) -> str:
    u = raw.upper()
    if u in ('EMERG', 'ALERT', 'CRIT', 'CRITICAL', 'FATAL', 'SEVERE', 'ERR'):
        return 'ERROR'
    if u in ('WARNING', 'NOTICE'):
        return 'WARN'
    if u == 'TRACE':
        return 'DEBUG'
    return u if u in ('INFO', 'WARN', 'ERROR', 'DEBUG') else 'INFO'


def _guess_service(message: str) -> str:
    m = message.lower()
    if 'auth' in m or 'login' in m or 'token' in m:
        return 'AuthService'
    if 'block' in m or 'replica' in m:
        return 'BlockManager'
    if 'pipeline' in m or 'stage' in m:
        return 'DataPipeline'
    if 'api' in m or 'endpoint' in m or 'request' in m:
        return 'APIGateway'
    if 'cache' in m or 'hit' in m or 'miss' in m:
        return 'CacheService'
    if 'database' in m or 'query' in m or 'sql' in m or 'connection pool' in m:
        return 'DatabaseService'
    if 'payment' in m or 'payment gateway' in m:
        return 'PaymentService'
    if 'notification' in m or 'email' in m:
        return 'NotificationService'
    if 'network' in m or 'timeout' in m or 'socket' in m:
        return 'NetworkService'
    return 'System'


def _extract_ip(text: str) -> str | None:
    m = IP_RE.search(text)
    return m.group(0) if m else None


def _extract_user(text: str) -> str | None:
    m = USER_ID_RE.search(text)
    return m.group(1) if m else None


def _extract_status(text: str) -> str | None:
    m = STATUS_CODE_RE.search(text) or HTTP_STATUS_RE.search(text)
    return m.group(1) if m else None


def detect_log_type(line: str) -> str:
    line = line.strip()
    if not line:
        return 'plain'
    if line.startswith('{'):
        try:
            json.loads(line)
            return 'json'
        except json.JSONDecodeError:
            pass
    if BGL_RE.match(line):
        return 'bgl'
    if TIMESTAMP_RE.match(line) and line.count(',') >= 3:
        return 'csv'
    if APACHE_NGINX_RE.match(line):
        return 'apache_nginx'
    if TIMESTAMP_RE.match(line) or LEVEL_RE.search(line):
        return 'application'
    return 'plain'


def parse_apache_nginx(line: str) -> ParsedLog | None:
    m = APACHE_NGINX_RE.match(line)
    if not m:
        return None
    _, remote_ip, _, timestamp, request, status, _ = m.groups()
    return ParsedLog(
        timestamp=timestamp,
        service='WebServer',
        log_level='INFO' if status and status.startswith('2') else 'ERROR',
        message=request or line,
        ip_address=remote_ip if remote_ip != '-' else None,
        user_id=None,
        status_code=status,
        raw=line,
        log_type='apache_nginx'
    )


def parse_json(line: str) -> ParsedLog | None:
    try:
        data = json.loads(line)
        ts = data.get('timestamp', data.get('time', data.get('@timestamp', '')))
        if isinstance(ts, (int, float)):
            from datetime import datetime
            ts = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        level = _normalize_level(str(
            data.get('level', data.get('log_level', data.get('severity', 'INFO')))
        ))
        msg = data.get('message', data.get('msg', data.get('event', str(data))))
        return ParsedLog(
            timestamp=str(ts)[:19] if ts else '',
            service=str(data.get('service', data.get('service_name', data.get('logger', _guess_service(str(msg)))))),
            log_level=level,
            message=str(msg),
            ip_address=data.get('ip', data.get('ip_address', _extract_ip(str(msg)))),
            user_id=data.get('user_id', data.get('userId', _extract_user(str(msg)))),
            status_code=str(data.get('status_code', data.get('status', ''))) or _extract_status(str(msg)),
            raw=line,
            log_type='json'
        )
    except (json.JSONDecodeError, TypeError):
        return None


def parse_bgl(line: str) -> ParsedLog | None:
    parts = line.split(maxsplit=9)
    if len(parts) < 10:
        return None
    label, _, date, _, event_time, _, subsystem, component, level, message = parts
    return ParsedLog(
        timestamp=f'{date} {event_time}',
        service=f'{subsystem}/{component}',
        log_level=_normalize_level(level),
        message=message,
        ip_address=None,
        user_id=None,
        status_code=None,
        raw=line,
        log_type='bgl',
        dataset_label=0 if label == '-' else 1,
    )


def parse_csv(line: str) -> ParsedLog | None:
    parts = [p.strip() for p in next(csv.reader(StringIO(line)), [])]
    if len(parts) < 4:
        return None
    ts = parts[0] if len(parts) > 0 else ''
    svc = parts[1] if len(parts) > 1 else 'System'
    lvl = _normalize_level(parts[2]) if len(parts) > 2 else 'INFO'
    msg = parts[3] if len(parts) > 3 else line
    return ParsedLog(
        timestamp=ts[:19] if ts else '',
        service=svc,
        log_level=lvl,
        message=msg,
        ip_address=_extract_ip(line),
        user_id=_extract_user(line),
        status_code=_extract_status(line),
        raw=line,
        log_type='csv'
    )


def parse_application(line: str) -> ParsedLog:
    ts_match = TIMESTAMP_RE.match(line)
    timestamp_match = ts_match or ANY_TIMESTAMP_RE.search(line)
    timestamp = timestamp_match.group('timestamp').replace('T', ' ')[:19] if timestamp_match else ''
    remainder = line[ts_match.end():] if ts_match else line
    lvl_match = LEVEL_RE.search(remainder)
    level = _normalize_level(lvl_match.group(1)) if lvl_match else 'INFO'
    message = remainder
    if lvl_match:
        message = remainder.replace(lvl_match.group(0), '', 1)
    message = re.sub(r'^[\s\-:|]+', '', message).strip() or line

    svc_match = re.match(r'^\[([^\]]+)\]\s*(.*)', message)
    service = None
    if svc_match:
        service = svc_match.group(1)
        message = svc_match.group(2)
    if not service:
        service = _guess_service(message)

    return ParsedLog(
        timestamp=timestamp,
        service=service,
        log_level=level,
        message=message,
        ip_address=_extract_ip(line),
        user_id=_extract_user(line),
        status_code=_extract_status(line),
        raw=line,
        log_type='application'
    )


def parse_plain(line: str) -> ParsedLog:
    ts = ANY_TIMESTAMP_RE.search(line)
    timestamp = ts.group('timestamp').replace('T', ' ')[:19] if ts else ''
    lvl = LEVEL_RE.search(line)
    level = _normalize_level(lvl.group(1)) if lvl else 'INFO'
    message = line.strip()
    return ParsedLog(
        timestamp=timestamp,
        service=_guess_service(message),
        log_level=level,
        message=message,
        ip_address=_extract_ip(line),
        user_id=_extract_user(line),
        status_code=_extract_status(line),
        raw=line,
        log_type='plain'
    )


def parse_line(line: str) -> ParsedLog | None:
    line = line.strip()
    if not line:
        return None
    log_type = detect_log_type(line)
    if log_type == 'apache_nginx':
        return parse_apache_nginx(line)
    if log_type == 'json':
        return parse_json(line)
    if log_type == 'bgl':
        return parse_bgl(line)
    if log_type == 'csv':
        return parse_csv(line)
    if log_type == 'application':
        return parse_application(line)
    return parse_plain(line)


def parse_logs(content: str) -> list[dict[str, Any]]:
    lines = []
    if '\n' not in content and len(content) < 260:
        try:
            with open(content, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except (OSError, IOError):
            pass
    if not lines:
        lines = content.splitlines()
    results = []
    for i, line in enumerate(lines):
        parsed = parse_line(line)
        if parsed:
            d = parsed.to_dict()
            d['id'] = f'log-{i}'
            results.append(d)
    return results


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/log_example.txt'
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        logs = parse_logs(f.read())
    for log in logs[:5]:
        print(log)
