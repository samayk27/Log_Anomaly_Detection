import { useState, useCallback, useEffect } from "react";
import { Sidebar } from "@/components/dashboard/Sidebar";
import { OverviewTab } from "@/components/dashboard/OverviewTab";
import { LogViewer } from "@/components/dashboard/LogViewer";
import { PipelineView } from "@/components/dashboard/PipelineView";
import { AnomalyDetection } from "@/components/dashboard/AnomalyDetection";
import { FeatureTable } from "@/components/dashboard/FeatureTable";
import { FileUpload } from "@/components/dashboard/FileUpload";
import { DatasetBrowser } from "@/components/dashboard/DatasetBrowser";
import { SystemMetrics } from "@/components/dashboard/SystemMetrics";
import { PipelineResult } from "@/lib/logParser";
import {
  fetchLogs,
  fetchStats,
  fetchRecentAnomalies,
  fetchFeatures,
  fetchDatasets,
  ingestLogs,
  checkBackend,
  type ApiLogEntry,
  type StatsResponse,
  type RecentAnomaly,
  type DatasetInventory,
} from "@/api/logApi";
import { LogEntry, AnomalyPoint, FeatureVector } from "@/types/dataTypes";
import { Clock, Database, Cpu } from "lucide-react";

function apiLogToLogEntry(api: ApiLogEntry, i: number): LogEntry {
  const level = (api.log_level?.toUpperCase() || "INFO") as LogEntry["level"];
  const validLevels: LogEntry["level"][] = ["INFO", "WARN", "ERROR", "DEBUG"];
  const lvl = validLevels.includes(level) ? level : "INFO";
  return {
    id: api.id ?? `api-${i}`,
    timestamp: api.timestamp ?? "",
    level: lvl,
    raw: api.message ?? api.raw ?? "",
    template: api.template ?? api.message ?? "",
    params: api.params ?? [],
    component: api.service ?? api.component ?? "System",
    prediction: api.prediction as "Normal" | "Anomaly" | undefined,
    confidence: api.confidence,
    source: api.source,
    detection_reason: api.detection_reason,
  };
}

function apiAnomalyToPoint(a: RecentAnomaly, i: number): AnomalyPoint {
  return {
    time: a.timestamp ?? `#${i + 1}`,
    anomalyScore: a.confidence ?? 0,
    isAnomaly: a.prediction === "Anomaly",
    hybridDecision: a.prediction === "Anomaly" ? "anomaly" : "normal",
    message: a.message,
    source: a.source,
    detection_reason: a.detection_reason,
    service: a.service,
    log_level: a.log_level,
    errorCount: (a as any).errorCount ?? 0,
    avgResponseTime: (a as any).avgResponseTime ?? 200,
    uniqueTemplates: (a as any).uniqueTemplates ?? 1,
    isolationForest: a.prediction === "Anomaly" ? "anomaly" : "normal",
    statistical: (a as any).errorCount > 5 ? "anomaly" : "normal",
  };
}

const Index = () => {
  const [activeTab, setActiveTab] = useState("overview");
  const [dataSource, setDataSource] = useState<"uploaded" | "backend">(
    "uploaded",
  );
  const [uploadedResult, setUploadedResult] = useState<PipelineResult | null>(
    null,
  );
  const [backendAvailable, setBackendAvailable] = useState(false);
  const [backendLogs, setBackendLogs] = useState<LogEntry[]>([]);
  const [backendStats, setBackendStats] = useState<StatsResponse | null>(null);
  const [backendAnomalies, setBackendAnomalies] = useState<AnomalyPoint[]>([]);
  const [backendFeatures, setBackendFeatures] = useState<FeatureVector[]>([]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [liveRefresh, setLiveRefresh] = useState(false);
  const [datasetInventory, setDatasetInventory] = useState<DatasetInventory | null>(null);

  const refetchBackend = useCallback(async () => {
    setBackendLoading(true);
    try {
      const [logsRes, statsRes, anomaliesRes, featuresRes] = await Promise.all([
        fetchLogs(),
        fetchStats(),
        fetchRecentAnomalies(),
        fetchFeatures(),
      ]);
      setBackendLogs((logsRes.logs ?? []).map(apiLogToLogEntry));
      setBackendStats(statsRes);
      setBackendAnomalies(
        (anomaliesRes.anomalies ?? []).map(apiAnomalyToPoint),
      );
      setBackendFeatures(featuresRes.features ?? []);
    } catch {
      setBackendLogs([]);
      setBackendStats(null);
      setBackendAnomalies([]);
      setBackendFeatures([]);
    } finally {
      setBackendLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!liveRefresh || !backendAvailable || dataSource !== "backend") return;
    const id = setInterval(refetchBackend, 5000);
    return () => clearInterval(id);
  }, [liveRefresh, backendAvailable, dataSource, refetchBackend]);

  useEffect(() => {
    checkBackend().then((ok) => {
      setBackendAvailable(ok);
      if (ok) {
        refetchBackend().then(() => setDataSource("backend"));
        fetchDatasets().then(setDatasetInventory).catch(() => setDatasetInventory(null));
      }
    });
  }, [refetchBackend]);

  const handlePipelineComplete = useCallback((result: PipelineResult) => {
    setUploadedResult(result);
    setDataSource("uploaded");
  }, []);

  const handleBackendIngest = useCallback(
    async (content: string) => {
      if (!backendAvailable) return;
      try {
        await ingestLogs(content);
        await refetchBackend();
        setDataSource("backend");
      } catch {
      }
    },
    [backendAvailable, refetchBackend],
  );

  const handleClearSystem = useCallback(() => {
    setUploadedResult(null);
    setDataSource("uploaded");
  }, []);

  const logs =
    dataSource === "backend" ? backendLogs : (uploadedResult?.logs ?? []);
  const timeSeriesData =
    dataSource === "backend"
      ? backendAnomalies
      : (uploadedResult?.anomalies ?? []);
  const featureVectors =
    dataSource === "backend"
      ? backendFeatures
      : (uploadedResult?.features ?? []);
  const apiStats = dataSource === "backend" ? backendStats : null;

  const tabTitles: Record<string, string> = {
    overview: "Dashboard",
    anomalies: "Threat Alerts",
    upload: "Import Data",
    system: "How It Works",
    logs: "Raw Data Explorer",
    pipeline: "Processing Steps",
    features: "ML Features",
  };

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-y-auto">
        <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-md px-8 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              {tabTitles[activeTab]}
            </h2>
            <p className="text-xs text-muted-foreground font-mono">
              Monitoring Active
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 bg-secondary rounded-md p-0.5">
              <button
                onClick={() => uploadedResult && setDataSource("uploaded")}
                disabled={!uploadedResult}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                  uploadedResult
                    ? "text-muted-foreground hover:text-foreground"
                    : "text-muted-foreground/40 cursor-not-allowed"
                }`}
              >
                <Database className="w-3 h-3 inline mr-1.5" />
                Uploaded
              </button>
              <button
                onClick={() => backendAvailable && setDataSource("backend")}
                disabled={!backendAvailable || backendLoading}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                  backendAvailable && !backendLoading
                    ? "text-muted-foreground hover:text-foreground"
                    : "text-muted-foreground/40 cursor-not-allowed"
                }`}
              >
                <Cpu className="w-3 h-3 inline mr-1.5" />
                ML Backend
              </button>
            </div>
            {backendAvailable && dataSource === "backend" && (
              <button
                onClick={() => setLiveRefresh(!liveRefresh)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                  liveRefresh
                    ? "bg-primary/20 text-primary"
                    : "bg-secondary text-muted-foreground hover:text-foreground"
                }`}
              >
                {liveRefresh ? "● Live" : "○ Live"}
              </button>
            )}
            <div className="flex items-center gap-2 text-xs text-success">
              <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
              System Online
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-secondary rounded-md text-xs text-muted-foreground">
              <Clock className="w-3 h-3" />
              {dataSource === "backend" ? "ML Backend" : "Uploaded"}
            </div>
          </div>
        </header>
        <div className="p-8">
          {activeTab === "upload" && (
            <div className="space-y-6">
              <FileUpload
                onPipelineComplete={handlePipelineComplete}
                onClearSystem={handleClearSystem}
                onBackendIngest={
                  backendAvailable ? handleBackendIngest : undefined
                }
              />
              <DatasetBrowser
                inventory={datasetInventory}
                onInventoryChange={setDatasetInventory}
                onAnalysisComplete={async () => {
                  await refetchBackend();
                  setDataSource("backend");
                }}
              />
            </div>
          )}
          {activeTab === "overview" && (
            <OverviewTab
              logs={logs}
              timeSeriesData={timeSeriesData}
              apiStats={apiStats}
            />
          )}
          {activeTab === "system" && <SystemMetrics />}
          {activeTab === "logs" && <LogViewer logs={logs} />}
          {activeTab === "pipeline" && <PipelineView />}
          {activeTab === "anomalies" && (
            <AnomalyDetection
              data={timeSeriesData}
              logs={logs}
              dataSource={dataSource}
            />
          )}
          {activeTab === "features" && <FeatureTable data={featureVectors} />}
        </div>
      </main>
    </div>
  );
};

export default Index;
