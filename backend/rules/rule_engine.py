import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List


@dataclass
class RuleResult:
    triggered: bool
    severity: str                     
    status: str                       
    rule_name: str
    message: str


                            
LOGIN_FAILURE_PATTERNS = [
    re.compile(r'authentication\s+failed', re.I),
    re.compile(r'login\s+failed', re.I),
    re.compile(r'invalid\s+credentials', re.I),
    re.compile(r'access\s+denied', re.I),
    re.compile(r'failed\s+login', re.I),
    re.compile(r'login\s+failure', re.I),
    re.compile(r'authentication\s+error', re.I),
    re.compile(r'bad\s+credentials', re.I),
    re.compile(r'wrong\s+password', re.I),
    re.compile(r'failed\s+authentication', re.I),
]
BRUTE_FORCE_PATTERNS = [
    re.compile(r'brute\s+force', re.I),
    re.compile(r'multiple\s+login\s+attempts', re.I),
    re.compile(r'too\s+many\s+failed\s+attempts', re.I),
    re.compile(r'brute\s+force\s+attack', re.I),
    re.compile(r'suspicious\s+login\s+activity', re.I),
]
DB_FAILURE_PATTERNS = [
    re.compile(r'database\s+connection\s+(?:failed|refused|timeout)', re.I),
    re.compile(r'connection\s+pool\s+exhausted', re.I),
    re.compile(r'db\s+(?:unreachable|connection)', re.I),
    re.compile(r'sql\s+(?:error|exception)', re.I),
    re.compile(r'connection\s+refused', re.I),
    re.compile(r'database\s+node\s+unreachable', re.I),
    re.compile(r'database\s+timeout', re.I),
    re.compile(r'db\s+connection\s+(?:failed|timeout|lost)', re.I),
]
UNAUTHORIZED_PATTERNS = [
    re.compile(r'unauthorized\s+access', re.I),
    re.compile(r'permission\s+denied', re.I),
    re.compile(r'access\s+denied', re.I),
    re.compile(r'access\s+denied\s+for\s+resource', re.I),
    re.compile(r'forbidden', re.I),
    re.compile(r'403\s+forbidden', re.I),
    re.compile(r'unauthorized\s+attempt', re.I),
]
CRITICAL_ERROR_PATTERNS = [
    re.compile(r'critical\s+', re.I),
    re.compile(r'fatal\s+', re.I),
    re.compile(r'service\s+(?:crashed|failed|error)', re.I),
    re.compile(r'system\s+(?:crashed|failed|error)', re.I),
    re.compile(r'application\s+(?:crashed|failed|error)', re.I),
    re.compile(r'out\s+of\s+memory', re.I),
    re.compile(r'stack\s+overflow', re.I),
    re.compile(r'memory\s+(?:corruption|leak)', re.I),
    re.compile(r'error.*crashed', re.I),
    re.compile(r'error.*failed', re.I),
    re.compile(r'.*crashed.*error', re.I),
    re.compile(r'.*crashed', re.I),
]
SUSPICIOUS_IP_PREFIXES = ('10.', '192.168.', '172.16.', '0.0.0.0')
KNOWN_BAD_PATTERNS = [
    re.compile(r'sql\s+injection', re.I),
    re.compile(r'xss\s+attempt', re.I),
    re.compile(r'path\s+traversal', re.I),
]


def _match_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def check_repeated_login_failures(
    logs: List[dict], current_idx: int, window: int = 10
) -> RuleResult:
    start = max(0, current_idx - window)
    slice_logs = logs[start:current_idx + 1]
    failures = sum(
        1 for log in slice_logs
        if _match_any(
            (log.get('message') or '') + ' ' + (log.get('raw') or ''),
            LOGIN_FAILURE_PATTERNS
        )
    )
    if failures >= 3:
        return RuleResult(True, 'HIGH', 'Anomaly', 'repeated_login_failures',
                          f'Multiple login failures ({failures}) detected')
    return RuleResult(False, 'LOW', 'Normal', 'repeated_login_failures', '')


def check_login_failure(message: str) -> RuleResult:
    text = (message or '').lower()
    if _match_any(text, LOGIN_FAILURE_PATTERNS):
        return RuleResult(True, 'HIGH', 'Anomaly', 'login_failure',
                          'Login failure detected')
    return RuleResult(False, 'LOW', 'Normal', 'login_failure', '')


def check_brute_force(message: str) -> RuleResult:
    text = (message or '').lower()
    if _match_any(text, BRUTE_FORCE_PATTERNS):
        return RuleResult(True, 'HIGH', 'Anomaly', 'brute_force',
                          'Brute force attempt detected')
    return RuleResult(False, 'LOW', 'Normal', 'brute_force', '')


def check_db_connection_failure(message: str) -> RuleResult:
    text = (message or '').lower()
    if _match_any(text, DB_FAILURE_PATTERNS):
        return RuleResult(True, 'HIGH', 'Anomaly', 'database_connection_failure',
                          'Database connection failure detected')
    return RuleResult(False, 'LOW', 'Normal', 'database_connection_failure', '')


def check_unauthorized_access(message: str) -> RuleResult:
    text = (message or '').lower()
    if _match_any(text, UNAUTHORIZED_PATTERNS):
        return RuleResult(True, 'HIGH', 'Anomaly', 'unauthorized_access',
                          'Unauthorized access detected')
    return RuleResult(False, 'LOW', 'Normal', 'unauthorized_access', '')


def check_unusual_request_frequency(
    logs: List[dict], current_idx: int, threshold: int = 50
) -> RuleResult:
    window = 20
    start = max(0, current_idx - window)
    count = len(logs[start:current_idx + 1])
    if count > threshold:
        return RuleResult(True, 'HIGH', 'Anomaly', 'unusual_request_frequency',
                          f'Unusual request frequency: {count} in window')
    return RuleResult(False, 'LOW', 'Normal', 'unusual_request_frequency', '')


def check_critical_errors(message: str) -> RuleResult:
    text = (message or '').lower()
    if _match_any(text, CRITICAL_ERROR_PATTERNS):
        return RuleResult(True, 'HIGH', 'Anomaly', 'critical_error',
                          'Critical error detected')
    return RuleResult(False, 'LOW', 'Normal', 'critical_error', '')


def check_suspicious_ip(log: dict) -> RuleResult:
    ip = log.get('ip_address') or ''
    msg = (log.get('message') or '') + (log.get('raw') or '')
    if not ip:
        ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', msg)
        ip = ip_match.group(0) if ip_match else ''
    if ip and any(msg.lower().count(k) for k in ('auth', 'login', 'failed')):
        if ip.startswith(SUSPICIOUS_IP_PREFIXES):
            return RuleResult(True, 'MEDIUM', 'Anomaly', 'suspicious_ip',
                              f'Suspicious internal IP in auth context: {ip}')
    if _match_any(msg, KNOWN_BAD_PATTERNS):
        return RuleResult(True, 'HIGH', 'Anomaly', 'known_attack_pattern',
                          'Known attack pattern detected')
    return RuleResult(False, 'LOW', 'Normal', 'suspicious_ip', '')


def run_rules_single(log: dict, all_logs: List[dict], idx: int) -> RuleResult | None:
    message = log.get('message') or log.get('raw') or ''
    checks = [
        check_critical_errors(message),
        check_login_failure(message),
        check_brute_force(message),
        check_db_connection_failure(message),
        check_unauthorized_access(message),
        check_suspicious_ip(log),
        check_repeated_login_failures(all_logs, idx),
        check_unusual_request_frequency(all_logs, idx, threshold=30),
    ]
    for r in checks:
        if r.triggered:
            return r
    return None


def run_rules_batch(logs: List[dict]) -> List[RuleResult | None]:
    return [run_rules_single(log, logs, i) for i, log in enumerate(logs)]


def combine_with_ml(rule_result: RuleResult | None, ml_prediction: str) -> tuple[str, str, str]:
    rule_triggered = rule_result and rule_result.triggered
    ml_anomaly = ml_prediction == 'Anomaly'
    reason = ''
    if rule_triggered and rule_result:
        reason = rule_result.message or rule_result.rule_name
    elif ml_anomaly:
        reason = 'ML anomaly score threshold exceeded'
    else:
        reason = 'Normal'
    if rule_triggered and ml_anomaly:
        return 'Anomaly', 'ML + Rules', reason
    if rule_triggered:
        return 'Anomaly', 'Rules', reason
    if ml_anomaly:
        return 'Anomaly', 'ML', reason
    return 'Normal', 'ML', reason


class RuleEngine:
    
    def __init__(self):
        self.rules_cache = {}
    
    def evaluate_log(self, log_record: dict) -> List[str]:
        triggered_rules = []
        
        message = log_record.get('message', '') + log_record.get('raw_message', '')
        
                                
        checks = [
            check_brute_force(message),
            check_db_connection_failure(message),
            check_unauthorized_access(message),
            check_suspicious_ip(log_record),
        ]
        
        for check in checks:
            if check.triggered:
                triggered_rules.append(check.rule_name)
        
        return triggered_rules
    
    def evaluate_batch(self, logs: List[dict]) -> List[List[str]]:
        return [self.evaluate_log(log) for log in logs]
