import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import hashlib

class AdvancedFeatureExtractor:
    
    def __init__(self, window_size_minutes: int = 5, sequence_length: int = 5):
        self.window_size_minutes = window_size_minutes
        self.sequence_length = sequence_length
        self.log_levels = ['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL', 'FATAL']
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.user_pattern = re.compile(r'user\s+([a-zA-Z0-9_]+)', re.IGNORECASE)
        self.status_pattern = re.compile(r'\b[1-5]\d{2}\b')  
        self.time_pattern = re.compile(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}')
        
    def parse_log_entry(self, log_line: str) -> Dict:
        parsed = {
            'raw_message': log_line.strip(),
            'timestamp': None,
            'log_level': None,
            'ip_address': None,
            'user': None,
            'status_code': None,
            'message_content': log_line.strip()
        }
        
        
        time_match = self.time_pattern.search(log_line)
        if time_match:
            try:
                parsed['timestamp'] = datetime.strptime(time_match.group(), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        
        
        for level in self.log_levels:
            if level in log_line.upper():
                parsed['log_level'] = level
                break
        
        
        ip_match = self.ip_pattern.search(log_line)
        if ip_match:
            parsed['ip_address'] = ip_match.group()
        
        
        user_match = self.user_pattern.search(log_line)
        if user_match:
            parsed['user'] = user_match.group(1)
        
        
        status_match = self.status_pattern.search(log_line)
        if status_match:
            parsed['status_code'] = int(status_match.group())
        
        
        content = log_line.strip()
        if time_match:
            content = content.replace(time_match.group(), '').strip()
        if parsed['log_level']:
            content = content.replace(parsed['log_level'], '').strip()
        parsed['message_content'] = content
        
        return parsed
    
    def extract_temporal_features(self, parsed_logs: List[Dict]) -> np.ndarray:
        n_logs = len(parsed_logs)
        features = np.zeros((n_logs, 6))  
        
        
        logs_with_time = [log for log in parsed_logs if log['timestamp'] is not None]
        logs_with_time.sort(key=lambda x: x['timestamp'])
        
        for i, log in enumerate(logs_with_time):
            current_time = log['timestamp']
            
            
            if i > 0:
                prev_time = logs_with_time[i-1]['timestamp']
                features[i, 0] = (current_time - prev_time).total_seconds()
            
            
            if i < n_logs - 1:
                next_time = logs_with_time[i+1]['timestamp']
                features[i, 1] = (next_time - current_time).total_seconds()
            
            
            hour = current_time.hour
            features[i, 2] = np.sin(2 * np.pi * hour / 24)
            features[i, 3] = np.cos(2 * np.pi * hour / 24)
            
            
            day = current_time.weekday()
            features[i, 4] = np.sin(2 * np.pi * day / 7)
            features[i, 5] = np.cos(2 * np.pi * day / 7)
        
        return features
    
    def extract_frequency_features(self, parsed_logs: List[Dict]) -> np.ndarray:
        n_logs = len(parsed_logs)
        features = np.zeros((n_logs, 8))  
        
        
        logs_with_time = [log for log in parsed_logs if log['timestamp'] is not None]
        logs_with_time.sort(key=lambda x: x['timestamp'])
        
        for i, log in enumerate(logs_with_time):
            current_time = log['timestamp']
            window_start = current_time - timedelta(minutes=self.window_size_minutes)
            window_end = current_time + timedelta(minutes=self.window_size_minutes)
            
            
            logs_in_window = [l for l in logs_with_time 
                            if window_start <= l['timestamp'] <= window_end]
            
            
            features[i, 0] = len(logs_in_window)
            
            
            for j, level in enumerate(self.log_levels[:4]):  
                level_count = sum(1 for l in logs_in_window if l['log_level'] == level)
                features[i, j+1] = level_count
            
            
            error_count = sum(1 for l in logs_in_window 
                            if l['log_level'] in ['ERROR', 'CRITICAL', 'FATAL'])
            features[i, 5] = error_count / max(len(logs_in_window), 1)
            
            
            unique_ips = set(l['ip_address'] for l in logs_in_window if l['ip_address'])
            features[i, 6] = len(unique_ips)
            
            
            unique_users = set(l['user'] for l in logs_in_window if l['user'])
            features[i, 7] = len(unique_users)
        
        return features
    
    def extract_structural_features(self, parsed_logs: List[Dict]) -> np.ndarray:
        n_logs = len(parsed_logs)
        features = np.zeros((n_logs, 10))  
        
        for i, log in enumerate(parsed_logs):
            message = log['message_content']
            
            
            features[i, 0] = len(message)
            
            
            features[i, 1] = len(message.split())
            
            
            features[i, 2] = len(re.findall(r'\b\d+\b', message))
            
            
            features[i, 3] = len(re.findall(r'[^a-zA-Z0-9\s]', message))
            
            
            if len(message) > 0:
                features[i, 4] = sum(1 for c in message if c.isupper()) / len(message)
            
            
            level_encoding = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3, 'CRITICAL': 4, 'FATAL': 5}
            if log['log_level']:
                features[i, 5] = level_encoding.get(log['log_level'], 1)
            
            
            if log['status_code']:
                features[i, 6] = log['status_code'] / 1000  
                
                features[i, 7] = 1 if log['status_code'] >= 400 else 0
            else:
                features[i, 6] = 0
                features[i, 7] = 0
            
            
            if log['ip_address']:
                ip_hash = int(hashlib.md5(log['ip_address'].encode()).hexdigest()[:8], 16)
                features[i, 8] = ip_hash / 1e8  
            else:
                features[i, 8] = 0
            
            
            msg_hash = int(hashlib.md5(message.encode()).hexdigest()[:8], 16)
            features[i, 9] = msg_hash / 1e8  
        
        return features
    
    def create_sequence_features(self, parsed_logs: List[Dict]) -> np.ndarray:
        n_logs = len(parsed_logs)
        sequence_features = []
        
        for i in range(n_logs):
            
            start_idx = max(0, i - self.sequence_length + 1)
            end_idx = i + 1
            sequence = parsed_logs[start_idx:end_idx]
            
            
            seq_features = []
            
            
            levels_in_seq = [log.get('log_level', 'INFO') for log in sequence]
            level_changes = sum(1 for j in range(1, len(levels_in_seq)) 
                              if levels_in_seq[j] != levels_in_seq[j-1])
            seq_features.append(level_changes / max(len(sequence) - 1, 1))
            
            
            error_count = sum(1 for log in sequence 
                           if log.get('log_level') in ['ERROR', 'CRITICAL', 'FATAL'])
            seq_features.append(error_count / len(sequence))
            
            
            timestamps = [log['timestamp'] for log in sequence if log['timestamp']]
            if len(timestamps) > 1:
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                seq_features.append(time_span / 60)  
            else:
                seq_features.append(0)
            
            
            lengths = [len(log['message_content']) for log in sequence]
            if len(lengths) > 1:
                seq_features.append(np.var(lengths) / 1000)  
            else:
                seq_features.append(0)
            
            
            ips = [log['ip_address'] for log in sequence if log['ip_address']]
            unique_ips = set(ips)
            seq_features.append(len(unique_ips) / max(len(ips), 1))
            
            
            while len(seq_features) < 10:
                seq_features.append(0)
            
            sequence_features.append(seq_features[:10])
        
        return np.array(sequence_features)
    
    def extract_all_features(self, log_lines: List[str]) -> np.ndarray:
        print("Parsing log entries...")
        parsed_logs = [self.parse_log_entry(line) for line in log_lines]
        
        print("Extracting temporal features...")
        temporal_features = self.extract_temporal_features(parsed_logs)
        
        print("Extracting frequency features...")
        frequency_features = self.extract_frequency_features(parsed_logs)
        
        print("Extracting structural features...")
        structural_features = self.extract_structural_features(parsed_logs)
        
        print("Creating sequence features...")
        sequence_features = self.create_sequence_features(parsed_logs)
        
        
        print("Combining features...")
        all_features = np.hstack([
            temporal_features,
            frequency_features,
            structural_features,
            sequence_features
        ])
        
        print(f"Feature extraction complete. Shape: {all_features.shape}")
        return all_features, parsed_logs
