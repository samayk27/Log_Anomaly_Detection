import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, CheckCircle2, AlertTriangle, Loader2, X, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { runPipeline, PipelineResult } from '@/lib/logParser';

interface FileUploadProps {
  onPipelineComplete: (result: PipelineResult) => void;
  onClearSystem?: () => void;
  /** When backend is available, send ingested content for ML analysis */
  onBackendIngest?: (content: string) => Promise<void>;
}

const ACCEPTED_EXTENSIONS = ['.log', '.txt', '.csv'];
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB

type HistoryEntry = {
  id: string;
  fileName: string;
  timestamp: string;
  result: PipelineResult;
};

const HISTORY_KEY = 'log_anomaly_upload_history_v1';

export function FileUpload({ onPipelineComplete, onClearSystem, onBackendIngest }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [pipelineStage, setPipelineStage] = useState(0);
  const [backendIngested, setBackendIngested] = useState<boolean | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // history helpers
  const loadHistory = (): HistoryEntry[] => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return [];
      return JSON.parse(raw) as HistoryEntry[];
    } catch {
      return [];
    }
  };

  const saveHistory = (items: HistoryEntry[]) => {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch {}
  };

  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());

  const stages = [
    'Validating file...',
    'Parsing log entries...',
    'Extracting templates...',
    'Computing feature vectors...',
    'Running anomaly detection...',
    'Generating hybrid decisions...',
  ];

  const processFile = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setBackendIngested(null);
    setFileName(file.name);

    // Validate extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Invalid file type. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`);
      return;
    }

    // Validate size
    if (file.size > MAX_FILE_SIZE) {
      setError('File exceeds 20MB limit.');
      return;
    }

    setProcessing(true);

    // Simulate pipeline stages for UX
    for (let i = 0; i < stages.length; i++) {
      setPipelineStage(i);
      await new Promise(r => setTimeout(r, 400 + Math.random() * 300));
    }

    try {
      const content = await file.text();
      const pipelineResult = runPipeline(content);

      if (pipelineResult.logs.length === 0) {
        setError('No parseable log entries found. Ensure the file contains timestamped log lines.');
        setProcessing(false);
        return;
      }

      setResult(pipelineResult);
      onPipelineComplete(pipelineResult);

      if (onBackendIngest) {
        try {
          await onBackendIngest(content);
          setBackendIngested(true);
        } catch {
          // Client pipeline already ran; backend ingest failed (e.g. server down)
          setBackendIngested(false);
        }
      }

      // persist to history (most-recent-first)
      try {
        const entry: HistoryEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          fileName: file.name,
          timestamp: new Date().toISOString(),
          result: pipelineResult,
        };
        const next = [entry, ...loadHistory()];
        setHistory(next);
        saveHistory(next);
      } catch {}
    } catch (err) {
      setError('Failed to process log file. Please check format.');
    } finally {
      setProcessing(false);
      setPipelineStage(0);
    }
  }, [onPipelineComplete, onBackendIngest]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const reset = () => {
    setResult(null);
    setError(null);
    setFileName('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeHistory = (id: string) => {
    const next = history.filter(h => h.id !== id);
    setHistory(next);
    saveHistory(next);
  };

  const clearHistory = () => {
    setHistory([]);
    saveHistory([]);
  };

  const loadEntry = (entry: HistoryEntry) => {
    setResult(entry.result);
    onPipelineComplete(entry.result);
    setFileName(entry.fileName);
  };

  const removeCurrentAndClear = () => {
    // remove possible matches by fileName or parsedLogs
    if (result) {
      const found = history.find(h => h.fileName === fileName || h.result.stats.parsedLogs === result.stats.parsedLogs);
      if (found) removeHistory(found.id);
    }
    reset();
    onClearSystem && onClearSystem();
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground mb-2">Log File Upload</h2>
        <p className="text-sm text-muted-foreground">
          Upload a <code className="text-primary">.log</code> or <code className="text-primary">.txt</code> file to run the full anomaly detection pipeline
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !processing && fileInputRef.current?.click()}
        className={cn(
          'relative rounded-xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300',
          isDragging
            ? 'border-primary bg-primary/5 scale-[1.01]'
            : 'border-border hover:border-primary/50 hover:bg-secondary/30',
          processing && 'pointer-events-none opacity-70'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".log,.txt,.csv"
          onChange={handleFileSelect}
          className="hidden"
        />

        <AnimatePresence mode="wait">
          {processing ? (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <Loader2 className="w-10 h-10 text-primary mx-auto animate-spin" />
              <p className="text-sm font-medium text-foreground">{stages[pipelineStage]}</p>
              <div className="max-w-xs mx-auto space-y-1">
                {stages.map((stage, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    {i < pipelineStage ? (
                      <CheckCircle2 className="w-3 h-3 text-success flex-shrink-0" />
                    ) : i === pipelineStage ? (
                      <Loader2 className="w-3 h-3 text-primary animate-spin flex-shrink-0" />
                    ) : (
                      <div className="w-3 h-3 rounded-full border border-border flex-shrink-0" />
                    )}
                    <span className={cn(
                      'font-mono',
                      i <= pipelineStage ? 'text-foreground' : 'text-muted-foreground'
                    )}>{stage}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              <Upload className={cn('w-10 h-10 mx-auto', isDragging ? 'text-primary' : 'text-muted-foreground')} />
              <div>
                <p className="text-sm font-medium text-foreground">
                  {isDragging ? 'Drop your log file here' : 'Drag & drop a log file, or click to browse'}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Supports .log, .txt, .csv • Max 20MB
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/30"
        >
          <AlertTriangle className="w-4 h-4 text-destructive flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
          <button onClick={reset} className="ml-auto">
            <X className="w-4 h-4 text-muted-foreground hover:text-foreground" />
          </button>
        </motion.div>
      )}

      {/* Result summary */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <div className="flex items-center gap-3 p-4 rounded-lg bg-success/10 border border-success/30">
            <CheckCircle2 className="w-5 h-5 text-success" />
            <div className="flex-1">
              <p className="text-sm font-medium text-foreground">Pipeline Complete — {fileName}</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {backendIngested
                  ? 'ML backend analysis complete. View Overview, Logs, and Anomalies for results.'
                  : onBackendIngest
                  ? 'File parsed successfully. Showing uploaded results; backend sync unavailable for this run.'
                  : 'All stages executed successfully. Data loaded into dashboard.'}
              </p>
            </div>
            <button
              onClick={reset}
              className="text-xs px-3 py-1.5 rounded-md bg-secondary text-muted-foreground hover:text-foreground transition-colors"
            >
              Upload Another
            </button>
            <button
              onClick={removeCurrentAndClear}
              className="text-xs px-3 py-1.5 rounded-md bg-destructive/10 text-destructive hover:bg-destructive/20 ml-2"
            >
              Remove & Clear
            </button>
            <button
              onClick={() => { reset(); onClearSystem && onClearSystem(); }}
              className="text-xs px-3 py-1.5 rounded-md bg-secondary text-muted-foreground hover:text-foreground ml-2"
            >
              Clear System Detections
            </button>
          </div>

          <p className="text-xs text-muted-foreground flex items-center gap-2">
            <ArrowRight className="w-3 h-3" />
            Navigate to <span className="text-primary font-medium">Overview</span>, <span className="text-primary font-medium">Logs</span>, <span className="text-primary font-medium">Anomalies</span>, or <span className="text-primary font-medium">Features</span> to explore results.
          </p>
        </motion.div>
      )}

      {/* Pipeline Architecture Reference */}
      {!result && !processing && (
        <div className="glass-card rounded-lg border border-border p-5">
          <h3 className="text-sm font-semibold text-foreground mb-3">Pipeline Architecture</h3>
          <div className="flex items-center gap-2 flex-wrap text-[10px] font-mono">
            {['Upload', 'Parse', 'Template Extract', 'Feature Vectors', 'Isolation Forest', 'Statistical', 'Hybrid Decision', 'Dashboard'].map((step, i) => (
              <span key={step} className="flex items-center gap-2">
                <span className="px-2 py-1 rounded bg-secondary text-muted-foreground">{step}</span>
                {i < 7 && <ArrowRight className="w-3 h-3 text-border" />}
              </span>
            ))}
          </div>
        </div>
      )}
      {/* Upload history */}
      {history.length > 0 && (
        <div className="glass-card rounded-lg border border-border p-5">
          <h3 className="text-sm font-semibold text-foreground mb-3">Upload History</h3>
          <div className="space-y-2 text-xs font-mono">
            {history.map(h => (
              <div key={h.id} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="w-4 h-4 text-muted-foreground" />
                  <div>
                    <div className="text-foreground">{h.fileName}</div>
                    <div className="text-muted-foreground text-[11px]">{new Date(h.timestamp).toLocaleString()}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => loadEntry(h)} className="text-xs px-2 py-1 rounded bg-secondary">Load</button>
                  <button onClick={() => removeHistory(h.id)} className="text-xs px-2 py-1 rounded bg-destructive/10 text-destructive">Remove</button>
                </div>
              </div>
            ))}
            <div className="pt-3">
              <button onClick={clearHistory} className="text-xs px-3 py-1.5 rounded-md bg-secondary text-muted-foreground">Clear History</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
