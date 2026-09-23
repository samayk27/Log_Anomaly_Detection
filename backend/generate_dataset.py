import random
import json
from datetime import datetime, timedelta
from pathlib import Path

LOG_TEMPLATES = {
    'normal_info': [
        "User {user} logged in successfully from {ip}",
        "Database connection established to {db_server}",
        "API request processed: {method} {endpoint} - {status}",
        "File upload completed: {filename} ({size}MB)",
        "Backup completed successfully for {service}",
        "Cache cleared for {module}",
        "Service {service} started on port {port}",
        "Session created for user {user}",
        "Payment processed successfully: ${amount}",
        "Email sent to {email} - campaign {campaign}",
        "Data synchronization completed for {table}",
        "Health check passed for {component}",
        "Configuration updated for {service}",
        "User {user} logged out from {ip}",
        "Scheduled task {task} completed successfully"
    ],
    'normal_debug': [
        "Debug: Processing request ID {request_id}",
        "Debug: Memory usage: {memory_mb}MB",
        "Debug: Query executed in {time}ms: {query}",
        "Debug: Cache hit for key {cache_key}",
        "Debug: Validating token for user {user}",
        "Debug: Loading configuration from {config_file}",
        "Debug: Connection pool stats: active={active}, idle={idle}",
        "Debug: Rate limit check for {ip}: {count}/{limit}",
        "Debug: Serializing object of type {object_type}",
        "Debug: Deserializing JSON payload ({size} bytes)"
    ],
    'warning_normal': [
        "WARNING: High memory usage detected: {memory_mb}MB",
        "WARNING: Slow query detected: {query} took {time}ms",
        "WARNING: Disk space low on {partition}: {space_free}% remaining",
        "WARNING: Rate limit approaching for {ip}: {count}/{limit}",
        "WARNING: Large file upload detected: {filename} ({size}MB)",
        "WARNING: Deprecated API endpoint used: {endpoint}",
        "WARNING: Configuration {config} using default value",
        "WARNING: SSL certificate expires in {days} days"
    ],
    'anomaly_security': [
        "FAILED LOGIN: Invalid credentials for user {user} from {ip}",
        "SECURITY ALERT: Brute force attack detected from {ip}",
        "UNAUTHORIZED ACCESS: Attempt to access {resource} without permission",
        "SUSPICIOUS ACTIVITY: Multiple failed logins for user {user} from {ip}",
        "BLOCKED IP: {ip} blocked due to suspicious activity",
        "SECURITY BREACH: Potential SQL injection attempt detected",
        "MALICIOUS REQUEST: Suspicious pattern detected in {endpoint}",
        "INTRUSION DETECTED: Unusual access pattern from {ip}"
    ],
    'anomaly_system': [
        "CRITICAL: Database connection timeout after {timeout}s",
        "ERROR: Service {service} crashed with exit code {code}",
        "FATAL: Out of memory error in {process}",
        "CRITICAL: Disk I/O error on {device}",
        "ERROR: Network partition detected - cannot reach {server}",
        "FATAL: Stack overflow in {module}",
        "CRITICAL: Deadlock detected in database transaction",
        "ERROR: Memory corruption detected in {component}"
    ],
    'anomaly_performance': [
        "PERFORMANCE ALERT: Response time {response_time}ms for {endpoint}",
        "SLOWDOWN: Queue depth exceeded threshold: {queue_size}",
        "TIMEOUT: Request to {service} timed out after {timeout}s",
        "OVERLOAD: CPU usage at {cpu_percent}% for {duration}s",
        "BOTTLENECK: Thread pool exhausted - {waiting} requests waiting",
        "THROTTLED: API calls throttled due to high load",
        "CONGESTION: Network latency {latency}ms to {server}"
    ]
}

USERS = ['admin', 'john_doe', 'jane_smith', 'mike_wilson', 'sarah_jones', 'bob_brown', 'alice_white', 'charlie_davis', 'emma_martin', 'david_lee']
IP_ADDRESSES = ['192.168.1.100', '10.0.0.50', '172.16.0.25', '203.0.113.10', '198.51.100.20', '192.0.2.30', '10.1.1.100', '172.20.0.15']
SUSPICIOUS_IPS = ['185.220.101.182', '109.234.38.41', '176.123.45.67', '94.102.49.90', '89.248.172.16']
SERVICES = ['auth-service', 'payment-api', 'user-management', 'order-service', 'notification-service', 'analytics-engine', 'file-storage', 'email-service']
DATABASES = ['primary_db', 'user_db', 'order_db', 'analytics_db', 'cache_db']
ENDPOINTS = ['/api/users', '/api/orders', '/api/payments', '/api/auth/login', '/api/files/upload', '/api/notifications', '/api/analytics']
METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
STATUS_CODES = ['200', '201', '204', '400', '401', '403', '404', '500', '502', '503']

def generate_log_line(log_type=None):
    if log_type is None:
        weights = [0.4, 0.2, 0.15, 0.08, 0.08, 0.06, 0.03]
        log_type = random.choices(list(LOG_TEMPLATES.keys()), weights=weights)[0]
    
    templates = LOG_TEMPLATES[log_type]
    template = random.choice(templates)
    
    substitutions = {
        'user': random.choice(USERS),
        'ip': random.choice(SUSPICIOUS_IPS if 'attack' in log_type or 'suspicious' in log_type else IP_ADDRESSES),
        'db_server': random.choice(DATABASES),
        'method': random.choice(METHODS),
        'endpoint': random.choice(ENDPOINTS),
        'status': random.choice(STATUS_CODES),
        'filename': f"file_{random.randint(1000, 9999)}.txt",
        'size': round(random.uniform(0.1, 500.2), 1),
        'service': random.choice(SERVICES),
        'port': random.randint(3000, 9000),
        'module': random.choice(['auth', 'payment', 'user', 'order', 'notification']),
        'amount': random.randint(10, 10000),
        'email': f"user{random.randint(1, 1000)}@example.com",
        'campaign': f"campaign_{random.randint(1, 50)}",
        'table': random.choice(['users', 'orders', 'payments', 'sessions', 'logs']),
        'component': random.choice(['database', 'cache', 'queue', 'load_balancer']),
        'task': random.choice(['backup', 'cleanup', 'report', 'sync', 'index']),
        'request_id': f"req_{random.randint(100000, 999999)}",
        'memory_mb': random.randint(100, 8000),
        'time': random.randint(50, 5000),
        'query': random.choice(['SELECT * FROM users', 'UPDATE orders SET status', 'DELETE FROM sessions']),
        'cache_key': f"key_{random.randint(1, 1000)}",
        'active': random.randint(1, 50),
        'idle': random.randint(1, 100),
        'count': random.randint(1, 100),
        'limit': random.randint(50, 1000),
        'object_type': random.choice(['User', 'Order', 'Payment', 'Session']),
        'config_file': random.choice(['app.conf', 'db.conf', 'cache.conf']),
        'space_free': random.randint(5, 30),
        'days': random.randint(1, 90),
        'timeout': random.randint(5, 60),
        'code': random.choice(['1', '2', '139', '255']),
        'process': random.choice(['worker', 'scheduler', 'api-server', 'background-job']),
        'device': random.choice(['/dev/sda1', '/dev/sdb2', '/dev/nvme0n1']),
        'server': random.choice(['db-primary', 'cache-node', 'api-server-1', 'worker-2']),
        'response_time': random.randint(5000, 30000),
        'queue_size': random.randint(100, 1000),
        'cpu_percent': random.randint(85, 100),
        'duration': random.randint(60, 600),
        'waiting': random.randint(10, 100),
        'latency': random.randint(500, 5000)
    }
    
    try:
        log_line = template.format(**substitutions)
    except KeyError as e:
        log_line = template
    
    timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    if 'critical' in log_type or 'fatal' in log_type:
        level = 'CRITICAL'
    elif 'error' in log_type:
        level = 'ERROR'
    elif 'warning' in log_type:
        level = 'WARN'
    elif 'debug' in log_type:
        level = 'DEBUG'
    else:
        level = 'INFO'
    
    return f"{timestamp_str} {level} {log_line}"

def generate_dataset(num_lines=50000, output_file=None):
    if output_file is None:
        output_file = Path(__file__).parent / "comprehensive_logs_50k.txt"
    
    print(f"Generating {num_lines} log lines...")
    
    logs = []
    anomaly_count = 0
    
    for i in range(num_lines):
        if i % 1000 < 80:
            if i % 100 < 10:
                logs.append(generate_log_line('anomaly_security'))
                anomaly_count += 1
            elif i % 100 < 20:
                logs.append(generate_log_line('anomaly_system'))
                anomaly_count += 1
            elif i % 100 < 30:
                logs.append(generate_log_line('anomaly_performance'))
                anomaly_count += 1
        else:
            if i % 10 < 4:
                logs.append(generate_log_line('normal_info'))
            elif i % 10 < 7:
                logs.append(generate_log_line('normal_debug'))
            else:
                logs.append(generate_log_line('warning_normal'))
    
    random.shuffle(logs)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for log in logs:
            f.write(log + '\n')
    
    print(f"Dataset generated: {len(logs)} lines")
    print(f"Estimated anomalies: {anomaly_count} ({anomaly_count/len(logs)*100:.1f}%)")
    print(f"Output file: {output_file}")
    
    return output_file

if __name__ == "__main__":
    generate_dataset(50000)
