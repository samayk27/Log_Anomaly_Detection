/**
 * API client for Log Anomaly Detection backend.
 * Uses fetch to communicate with FastAPI backend.
 */

import { FeatureVector } from '@/types/dataTypes';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ApiLogEntry {
  id?: string;
  timestamp: string;
  service: string;
  log_level: string;
  message: string;
  raw?: string;
  prediction?: string;
  confidence?: number;
  source?: string;
  detection_reason?: string;
  component?: string;
  template?: string;
  params?: string[];
}

export interface StatsResponse {
  total_logs: number;
  anomaly_count: number;
  normal_count: number;
  anomaly_percentage: number;
}

export interface AnalyzeResponse {
  message: string;
  prediction: string;
  confidence: number;
  source: string;
}

export interface RecentAnomaly {
  message: string;
  prediction: string;
  confidence: number;
  source: string;
  timestamp?: string;
  detection_reason?: string;
  service?: string;
  log_level?: string;
}

export interface DatasetInfo {
  path: string;
  name: string;
  extension: string;
  size_bytes: number;
  records: number;
}

export interface DatasetInventory {
  datasets: DatasetInfo[];
  split: {
    available: boolean;
    train_path: string;
    test_path: string;
  };
}

export interface DatasetEvaluation {
  test_sources: string[];
  test_records: number;
  labeling: string;
  ml_only: { accuracy: number; precision: number; recall: number; f1_score: number };
  hybrid_ml_rules: { accuracy: number; precision: number; recall: number; f1_score: number };
}

export interface DatasetAnalysisResult {
  message: string;
  paths: string[];
  count: number;
  anomalies: number;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchLogs(): Promise<{ logs: ApiLogEntry[] }> {
  const res = await fetch(`${API_BASE}/logs`);
  return handleResponse(res);
}

export async function analyzeLog(message: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze-log`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  return handleResponse(res);
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/stats`);
  return handleResponse(res);
}

export async function fetchRecentAnomalies(): Promise<{ anomalies: RecentAnomaly[] }> {
  const res = await fetch(`${API_BASE}/recent-anomalies`);
  return handleResponse(res);
}

export async function fetchFeatures(): Promise<{ features: FeatureVector[] }> {
  const res = await fetch(`${API_BASE}/features`);
  return handleResponse(res);
}

export async function ingestLogs(content: string): Promise<{ count: number; anomalies: number }> {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  const data = await handleResponse<{ count: number; anomalies: number }>(res);
  return data;
}

/** Check if backend is reachable */
export async function checkBackend(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/stats`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchDatasets(): Promise<DatasetInventory> {
  const res = await fetch(`${API_BASE}/datasets`);
  return handleResponse(res);
}

export async function splitDatasets(testRatio = 0.2): Promise<DatasetInventory> {
  const res = await fetch(`${API_BASE}/datasets/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ test_ratio: testRatio, random_state: 42 }),
  });
  await handleResponse(res);
  return fetchDatasets();
}

export async function evaluateDatasets(): Promise<DatasetEvaluation> {
  const res = await fetch(`${API_BASE}/datasets/evaluate`, { method: "POST" });
  return handleResponse(res);
}

export async function trainOnSplit(): Promise<{ train_records: number }> {
  const res = await fetch(`${API_BASE}/datasets/train`, { method: "POST" });
  return handleResponse(res);
}

export async function analyzeDatasets(paths: string[]): Promise<DatasetAnalysisResult> {
  const res = await fetch(`${API_BASE}/datasets/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths }),
  });
  return handleResponse(res);
}
