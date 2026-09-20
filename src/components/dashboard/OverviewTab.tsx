import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { AlertTriangle, FileText, ShieldAlert, BarChart3 } from "lucide-react";
import { MetricCard } from "./MetricCard";
import { AnomalyChart } from "./AnomalyChart";
import { AnomalyPoint, LogEntry } from "@/types/dataTypes";
import type { StatsResponse } from "@/api/logApi";

interface OverviewTabProps {
  logs: LogEntry[];
  timeSeriesData: AnomalyPoint[];
  apiStats?: StatsResponse | null;
}

export function OverviewTab({
  logs,
  timeSeriesData,
  apiStats,
}: OverviewTabProps) {
  const serviceAnomalyData = useMemo(() => {
    const anomalyLogs = logs.filter((l) => l.prediction === "Anomaly");
    const byService = new Map<string, number>();
    anomalyLogs.forEach((l) => {
      const svc = l.component || "System";
      byService.set(svc, (byService.get(svc) ?? 0) + 1);
    });
    return Array.from(byService.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [logs]);

  const stats = useMemo(() => {
    if (apiStats) {
      return {
        total: apiStats.total_logs,
        errors: logs.filter((l) => l.level === "ERROR").length,
        anomalies: apiStats.anomaly_count,
        templates: new Set(logs.map((l) => l.template)).size,
        normal: apiStats.normal_count,
        anomalyPct: apiStats.anomaly_percentage,
      };
    }
    const errors = logs.filter((l) => l.level === "ERROR").length;
    const warnings = logs.filter((l) => l.level === "WARN").length;
    const anomalies = timeSeriesData.filter(
      (d) => d.hybridDecision === "anomaly",
    ).length;
    const templates = new Set(logs.map((l) => l.template)).size;
    return {
      errors,
      warnings,
      anomalies,
      templates,
      total: logs.length,
      normal: logs.length - anomalies,
      anomalyPct: 0,
    };
  }, [logs, timeSeriesData, apiStats]);

  const hasData = logs.length > 0;
  const criticalErrors = logs
    .filter((log) => log.level === "ERROR");

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="rounded-lg border border-border bg-muted/20 p-5 mb-6 flex items-start gap-4">
        <div className="bg-primary/20 p-3 rounded-full">
          <ShieldAlert className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h3 className="text-base font-semibold text-foreground mb-1">
            System Monitoring Active
          </h3>
          <p className="text-sm text-muted-foreground">
            {apiStats
              ? "Our AI-driven security engine is actively scanning your system in real-time. It automatically detects unusual behavior, potential security breaches, and critical system failures."
              : "Client-side scanning mode is active. Upload data to begin analyzing for potential threats and anomalies."}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Total Events Analyzed"
          value={stats.total.toLocaleString()}
          subtitle={
            hasData ? "From current data source" : "Upload data to start"
          }
          icon={FileText}
          variant="primary"
          progress={hasData ? 100 : 0}
        />
        <MetricCard
          title="Critical Errors"
          value={stats.errors}
          subtitle="System failures requiring attention"
          icon={AlertTriangle}
          variant="danger"
          progress={stats.total > 0 ? (stats.errors / stats.total) * 100 : 0}
        />
        <MetricCard
          title="Potential Threats"
          value={stats.anomalies}
          subtitle={
            apiStats && stats.total > 0
              ? `${(stats.anomalyPct ?? 0).toFixed(1)}% of all events`
              : "AI + Security Rules"
          }
          icon={ShieldAlert}
          variant="warning"
          progress={stats.total > 0 ? (stats.anomalies / stats.total) * 100 : 0}
        />
        <MetricCard
          title={apiStats ? "Safe Events" : "Event Types"}
          value={apiStats ? (stats.normal ?? 0) : stats.templates}
          subtitle={
            apiStats ? "Verified normal activity" : "Distinct system activities"
          }
          icon={BarChart3}
          variant={apiStats ? "success" : "default"}
          progress={stats.total > 0 ? ((apiStats ? (stats.normal ?? 0) : stats.templates) / stats.total) * 100 : 0}
        />
      </div>

      <div className="glass-card rounded-lg border border-destructive/30 p-6">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-medium text-foreground">Critical Errors</h3>
            <p className="text-xs text-muted-foreground">
              Error-level events requiring investigation
            </p>
          </div>
          <span className="rounded-md bg-destructive/10 px-2 py-1 text-xs font-semibold text-destructive">
            {stats.errors} total
          </span>
        </div>
        {criticalErrors.length > 0 ? (
          <div className="max-h-[32rem] overflow-y-auto divide-y divide-border rounded-lg border border-border">
            {criticalErrors.map((log) => (
              <div key={log.id} className="grid grid-cols-[auto_1fr_auto] items-start gap-3 px-3 py-3 text-xs">
                <AlertTriangle className="mt-0.5 h-4 w-4 text-destructive" />
                <div className="min-w-0">
                  <p className="truncate font-medium text-foreground">{log.raw}</p>
                  <p className="mt-1 text-muted-foreground">
                    {log.component || "System"} {log.detection_reason ? `· ${log.detection_reason}` : ""}
                  </p>
                </div>
                <time className="whitespace-nowrap font-mono text-muted-foreground">
                  {log.timestamp || "Unknown time"}
                </time>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-xs text-muted-foreground">
            No critical errors in the current dataset.
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {serviceAnomalyData.length > 0 && (
          <div className="glass-card rounded-lg border border-border p-6 lg:col-span-2">
            <div className="mb-4">
              <h3 className="text-sm font-medium text-foreground mb-1">
                Threats by System Area
              </h3>
              <p className="text-xs text-muted-foreground">
                Which parts of the system are most affected
              </p>
            </div>
            <div className="mb-4 bg-destructive/10 border border-destructive/20 rounded-lg p-3 text-xs">
              <div className="font-semibold text-destructive mb-1">What This Shows</div>
              <div className="text-muted-foreground">The bar chart displays which system components (like Database, API, Auth) have the most detected threats. Higher bars mean more issues in that area.</div>
            </div>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={serviceAnomalyData}
                  margin={{ top: 10, right: 10, bottom: 10, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(220, 14%, 18%)"
                  />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} width={35} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(220, 18%, 10%)",
                      border: "1px solid hsl(220, 14%, 18%)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar
                    dataKey="count"
                    fill="hsl(0, 72%, 55%)"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        <div className="glass-card rounded-lg border border-border p-6">
          <div className="mb-4">
            <h3 className="text-sm font-medium text-foreground mb-1">
              System Errors Over Time
            </h3>
            <p className="text-xs text-muted-foreground">
              When errors occurred in your system
            </p>
          </div>
          <div className="mb-4 bg-warning/10 border border-warning/20 rounded-lg p-3 text-xs">
            <div className="font-semibold text-warning mb-1">What This Shows</div>
            <div className="text-muted-foreground">This line chart tracks error counts over time. Spikes indicate periods with more system problems that may need investigation.</div>
          </div>
          <div className="h-[250px]">
            <AnomalyChart
              data={timeSeriesData}
              metric="errorCount"
              title=""
            />
          </div>
        </div>
        <div className="glass-card rounded-lg border border-border p-6">
          <div className="mb-4">
            <h3 className="text-sm font-medium text-foreground mb-1">
              System Response Time
            </h3>
            <p className="text-xs text-muted-foreground">
              How fast the system responds to requests
            </p>
          </div>
          <div className="mb-4 bg-primary/10 border border-primary/20 rounded-lg p-3 text-xs">
            <div className="font-semibold text-primary mb-1">What This Shows</div>
            <div className="text-muted-foreground">This chart shows response time in milliseconds. Lower values mean faster performance. High values may indicate system slowdowns.</div>
          </div>
          <div className="h-[250px]">
            <AnomalyChart
              data={timeSeriesData}
              metric="avgResponseTime"
              title=""
            />
          </div>
        </div>
      </div>

      <div className="glass-card rounded-lg border border-border p-6">
        <div className="mb-4">
          <h3 className="text-sm font-medium text-foreground mb-1">
            AI Threat Risk Score Over Time
          </h3>
          <p className="text-xs text-muted-foreground">
            How risky each time period was according to our AI
          </p>
        </div>
        <div className="mb-4 bg-purple-500/10 border border-purple-500/20 rounded-lg p-3 text-xs">
          <div className="font-semibold text-purple-500 mb-1">What This Shows</div>
          <div className="text-muted-foreground">The AI assigns a risk score (0-1) to each time period. Higher scores mean more suspicious activity detected. Values above 0.5 indicate potential threats.</div>
        </div>
        <div className="h-[300px]">
          <AnomalyChart
            data={timeSeriesData}
            metric="anomalyScore"
            title=""
          />
        </div>
      </div>
    </div>
  );
}
