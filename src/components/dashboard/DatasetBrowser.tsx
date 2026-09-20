import { useState } from "react";
import { Database, FileText, Loader2, Play, Scissors } from "lucide-react";
import {
  analyzeDatasets,
  evaluateDatasets,
  fetchDatasets,
  splitDatasets,
  trainOnSplit,
  type DatasetEvaluation,
  type DatasetInfo,
  type DatasetInventory,
} from "@/api/logApi";

interface DatasetBrowserProps {
  inventory: DatasetInventory | null;
  onInventoryChange: (inventory: DatasetInventory) => void;
  onAnalysisComplete: () => Promise<void>;
}

const formatBytes = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

export function DatasetBrowser({ inventory, onInventoryChange, onAnalysisComplete }: DatasetBrowserProps) {
  const [busy, setBusy] = useState<"split" | "train" | "evaluate" | "analyze" | null>(null);
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [evaluation, setEvaluation] = useState<DatasetEvaluation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSplit = async () => {
    setBusy("split");
    setError(null);
    try {
      onInventoryChange(await splitDatasets());
      setEvaluation(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to split datasets");
    } finally {
      setBusy(null);
    }
  };

  const runEvaluation = async () => {
    setBusy("evaluate");
    setError(null);
    try {
      setEvaluation(await evaluateDatasets());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to evaluate test data");
    } finally {
      setBusy(null);
    }
  };

  const runTraining = async () => {
    setBusy("train");
    setError(null);
    try {
      await trainOnSplit();
      onInventoryChange(await fetchDatasets());
      setEvaluation(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to train the model");
    } finally {
      setBusy(null);
    }
  };

  const runAnalysis = async () => {
    if (selectedPaths.length === 0) return;
    setBusy("analyze");
    setError(null);
    try {
      await analyzeDatasets(selectedPaths);
      await onAnalysisComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze selected test datasets");
    } finally {
      setBusy(null);
    }
  };

  const togglePath = (path: string) => {
    setSelectedPaths((current) => current.includes(path)
      ? current.filter((selected) => selected !== path)
      : [...current, path]);
  };

  const selectAll = () => {
    setSelectedPaths((current) => current.length === inventory.datasets.length
      ? []
      : inventory.datasets.map((dataset) => dataset.path));
  };

  if (!inventory) return null;

  return (
    <section className="rounded-xl border border-border bg-card/60 p-5 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="rounded-lg bg-primary/10 p-2"><Database className="h-5 w-5 text-primary" /></div>
          <div>
            <h3 className="font-semibold text-foreground">Available datasets</h3>
              <p className="text-xs text-muted-foreground">Recursive testing files under datasets/test/</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button onClick={runSplit} disabled={busy !== null} className="inline-flex items-center gap-2 rounded-md bg-secondary px-3 py-2 text-xs font-medium text-foreground disabled:opacity-50">
            {busy === "split" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Scissors className="h-3.5 w-3.5" />}
            Refresh test split
          </button>
          <button onClick={runTraining} disabled={busy !== null} className="inline-flex items-center gap-2 rounded-md bg-secondary px-3 py-2 text-xs font-medium text-foreground disabled:opacity-50">
            {busy === "train" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Database className="h-3.5 w-3.5" />}
            Train model
          </button>
          <button onClick={runEvaluation} disabled={busy !== null || !inventory.split.available} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-medium text-primary-foreground disabled:opacity-50">
            {busy === "evaluate" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Test split
          </button>
          <button onClick={runAnalysis} disabled={busy !== null || selectedPaths.length === 0} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-medium text-primary-foreground disabled:opacity-50">
            {busy === "analyze" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Analyze selected
          </button>
        </div>
      </div>
      {inventory.datasets.length > 0 && (
        <button onClick={selectAll} className="text-xs font-medium text-primary hover:underline">
          {selectedPaths.length === inventory.datasets.length ? "Clear selection" : "Select all test files"}
        </button>
      )}
      <div className="divide-y divide-border rounded-lg border border-border">
        {inventory.datasets.map((dataset: DatasetInfo) => (
          <label key={dataset.path} className="flex cursor-pointer items-center justify-between gap-3 px-3 py-2.5 text-xs hover:bg-secondary/40">
            <div className="flex min-w-0 items-center gap-2"><input type="checkbox" checked={selectedPaths.includes(dataset.path)} onChange={() => togglePath(dataset.path)} /><FileText className="h-4 w-4 shrink-0 text-muted-foreground" /><span className="truncate text-foreground">{dataset.path}</span></div>
            <span className="shrink-0 text-muted-foreground">{dataset.records.toLocaleString()} records · {formatBytes(dataset.size_bytes)}</span>
          </label>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">{inventory.split.available ? `Held-out files: ${inventory.split.test_path}` : "Train the model to create datasets/test/ from each source file."}</p>
      {error && <p className="text-xs text-destructive">{error}</p>}
      {evaluation && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          {["ml_only", "hybrid_ml_rules"].map((key) => {
            const metric = evaluation[key as "ml_only" | "hybrid_ml_rules"];
            return <div key={key} className="rounded-lg bg-secondary/60 p-3"><div className="mb-2 font-medium text-foreground">{key === "ml_only" ? "ML only" : "ML + rules"}</div><div className="text-muted-foreground">F1 {(metric.f1_score * 100).toFixed(1)}% · Precision {(metric.precision * 100).toFixed(1)}% · Recall {(metric.recall * 100).toFixed(1)}%</div></div>;
          })}
        </div>
      )}
    </section>
  );
}