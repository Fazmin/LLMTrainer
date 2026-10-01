import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";
import { backend } from "../backend";
import type { JobEvent } from "../bindings";
import { describeError, type ErrorCard } from "../content/errors";

/** What the progress bar shows for a running job. */
export interface JobView {
  jobId: string | null;
  label: string;
  fraction: number | null;
  message: string;
  unit: string;
  done: number;
  total: number | null;
  bytesPerSec: number | null;
  etaSeconds: number | null;
}

/**
 * Runs one background job at a time and turns its event stream into state for a progress bar. The first event carries
 * the job id, so the job can be cancelled while it runs.
 */
export function useJobRunner() {
  const qc = useQueryClient();
  const [job, setJob] = useState<JobView | null>(null);
  const [error, setError] = useState<ErrorCard | null>(null);
  const jobId = useRef<string | null>(null);

  const run = useCallback(
    async <T,>(label: string, start: (onEvent: (e: JobEvent) => void) => Promise<T>): Promise<T | null> => {
      setError(null);
      jobId.current = null;
      setJob({ jobId: null, label, fraction: null, message: label, unit: "", done: 0, total: null, bytesPerSec: null, etaSeconds: null });
      try {
        const result = await start((e) => {
          if (e.type === "state") {
            jobId.current = e.jobId;
            setJob((j) => (j ? { ...j, jobId: e.jobId } : j));
          } else {
            const p = e.progress;
            setJob((j) => ({
              jobId: e.jobId,
              label: j?.label ?? label,
              fraction: p.total && p.total > 0 ? Math.min(1, p.done / p.total) : null,
              message: p.message,
              unit: p.unit,
              done: p.done,
              total: p.total,
              bytesPerSec: p.bytesPerSec,
              etaSeconds: p.etaSeconds,
            }));
          }
        });
        return result;
      } catch (e) {
        if ((e as { kind?: string })?.kind !== "cancelled") setError(describeError(e));
        return null;
      } finally {
        setJob(null);
        jobId.current = null;
        void qc.invalidateQueries({ queryKey: ["datasets"] });
        void qc.invalidateQueries({ queryKey: ["dataset"] });
        void qc.invalidateQueries({ queryKey: ["recommendation"] });
      }
    },
    [qc],
  );

  const cancel = useCallback(() => {
    if (jobId.current) void backend.cancelJob(jobId.current);
  }, []);

  return { job, error, run, cancel, clearError: () => setError(null) };
}
