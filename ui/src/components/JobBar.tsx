import { X } from "lucide-react";
import type { JobView } from "../state/jobs";
import { fmtBytes, fmtCount, fmtRoughTime } from "../lib/format";
import { Button } from "./Button";

/** Progress of a background job: what it is doing, how far along, how fast, and a way to stop it. */
export function JobBar({ job, onCancel }: { job: JobView; onCancel: () => void }) {
  const pct = job.fraction != null ? Math.round(job.fraction * 100) : null;
  const bytes = job.unit === "bytes";
  return (
    <div role="status" aria-live="polite" className="rounded-[var(--radius-panel)] border border-hairline bg-surface px-5 py-4">
      <div className="flex items-center gap-4">
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate text-[15px] font-medium">{job.message}</p>
          <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-surface-2" aria-hidden>
            <div
              className={`h-full rounded-full bg-accent transition-[width] duration-300 ${pct == null ? "w-1/3 animate-pulse" : ""}`}
              style={pct != null ? { width: `${pct}%` } : undefined}
            />
          </div>
          <p className="mt-2 text-[13px] text-ink-2">
            {pct != null && <>{pct}%</>}
            {bytes && job.total != null && <> ({fmtBytes(job.done)} of {fmtBytes(job.total)})</>}
            {bytes && job.bytesPerSec != null && <> at {fmtBytes(job.bytesPerSec)} a second</>}
            {job.etaSeconds != null && job.etaSeconds > 1 && <>, {fmtRoughTime(job.etaSeconds)} left</>}
            {!bytes && pct == null && job.done > 0 && <>{fmtCount(job.done)} {job.unit} so far</>}
          </p>
        </div>
        <Button size="sm" icon={<X size={14} />} onClick={onCancel} disabled={!job.jobId}>Cancel</Button>
      </div>
    </div>
  );
}
