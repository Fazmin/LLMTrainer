import clsx from "clsx";
import { ArrowRight, Check, Loader2, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";
import { backend } from "../backend";
import { Button } from "../components/Button";
import { JobBar } from "../components/JobBar";
import { describeError, type ErrorCard } from "../content/errors";
import { useJobRunner } from "../state/jobs";
import { useHardware } from "../state/queries";

type Step = "intro" | "choose" | "working";
type Phase = "text" | "model" | "start";

const PHASES: { id: Phase; label: string }[] = [
  { id: "text", label: "Getting the text" },
  { id: "model", label: "Setting up a small model" },
  { id: "start", label: "Starting to read" },
];

/** First launch: what this is, whether this computer can do it, and one click to a first training. */
export function Welcome() {
  const navigate = useNavigate();
  const hw = useHardware();
  const { job, run, cancel } = useJobRunner();
  const [step, setStep] = useState<Step>("intro");
  const [phase, setPhase] = useState<Phase>("text");
  const [error, setError] = useState<ErrorCard | null>(null);
  const [usedOffline, setUsedOffline] = useState(false);

  const gpu = hw.data?.backends.find((b) => b.kind === hw.data?.selected);
  const cpuOnly = hw.data?.selected === "cpu";

  const quickStart = async (starterId: string) => {
    setStep("working");
    setError(null);
    setPhase("text");
    setUsedOffline(starterId === "sampler");
    const dataset = await run(starterId === "sampler" ? "Writing the sample text" : "Downloading simple stories", (cb) => backend.installStarter(starterId, cb));
    if (!dataset) {
      // Cancelled, or it failed (the job runner kept the reason). Let the person choose again.
      setError({ title: "That did not finish", body: "You can try again, or use the small sample that needs no internet." });
      return;
    }
    try {
      setPhase("model");
      const presets = await backend.presetConfigs();
      const tiny = presets.find((p) => p.preset === "tiny")!;
      const created = await backend.createRun({
        name: null,
        preset: "tiny",
        model: tiny.model,
        train: tiny.train,
        datasetId: dataset.summary.id,
        goal: { type: "minutes", value: 10 },
      });
      setPhase("start");
      await backend.startRun(created.id);
      navigate(`/train/${created.id}`, { replace: true });
    } catch (e) {
      setError(describeError(e));
    }
  };

  return (
    <div className="mx-auto flex min-h-full max-w-[720px] flex-col justify-center px-8 py-14">
      <img src="/app-icon.png" alt="" width={44} height={44} className="rounded-[10px]" />
      <h1 className="mt-6 text-[34px] font-semibold leading-tight tracking-tight">Teach a small computer program to write.</h1>

      {step === "intro" && (
        <>
          <p className="mt-5 max-w-[56ch] text-[17px] leading-relaxed text-ink-2">
            LLM Trainer builds a language model from nothing, on this computer. It starts out knowing no words at all. You give it text to read, and over the next few minutes you can watch it learn to spell, then to write.
          </p>
          <ul className="m-0 mt-8 list-none space-y-3 p-0 text-[15px]">
            <li className="flex gap-3"><Check size={18} className="mt-0.5 shrink-0 text-good-ink" aria-hidden /><span>Nothing leaves your computer except the one download of practice text.</span></li>
            <li className="flex gap-3"><Check size={18} className="mt-0.5 shrink-0 text-good-ink" aria-hidden /><span>You can stop any time and pick up later.</span></li>
            <li className="flex gap-3">
              {cpuOnly ? <TriangleAlert size={18} className="mt-0.5 shrink-0 text-warning-ink" aria-hidden /> : <Check size={18} className="mt-0.5 shrink-0 text-good-ink" aria-hidden />}
              <span>
                {hw.data
                  ? cpuOnly
                    ? `No graphics processor was found on this ${hw.data.cpu} computer, so training will be slow. The smallest model still works.`
                    : `This computer (${hw.data.cpu}, ${Math.round(hw.data.ramGb)} GB memory) can use ${gpu?.name ?? "its graphics processor"}, so training will be quick.`
                  : "Checking this computer…"}
              </span>
            </li>
          </ul>
          <Button className="mt-10 self-start" variant="primary" size="lg" icon={<ArrowRight size={16} />} onClick={() => setStep("choose")}>Get started</Button>
        </>
      )}

      {step === "choose" && (
        <>
          <p className="mt-5 text-[17px] text-ink-2">How would you like to begin?</p>
          <div className="mt-7 space-y-3">
            <button onClick={() => void quickStart("tinystories-quick")} className="w-full rounded-[var(--radius-panel)] border border-accent bg-surface p-5 text-left ring-1 ring-accent transition-colors hover:bg-accent-soft">
              <span className="flex items-center gap-2 text-[17px] font-semibold">Quick start <span className="rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent-ink">Recommended</span></span>
              <span className="mt-1.5 block text-[14px] leading-snug text-ink-2">
                Download a collection of simple stories (about 87 MB) and train a small model on it for ten minutes. You will see it go from nonsense to words to sentences.
              </span>
            </button>
            <button onClick={() => navigate("/data")} className="w-full rounded-[var(--radius-panel)] border border-hairline bg-surface p-5 text-left transition-colors hover:border-axis">
              <span className="text-[17px] font-semibold">Use my own text</span>
              <span className="mt-1.5 block text-[14px] leading-snug text-ink-2">Choose folders of notes, books or code. You pick the model size and how long to train.</span>
            </button>
            <button onClick={() => void quickStart("sampler")} className="w-full rounded-[var(--radius-panel)] border border-hairline bg-surface p-5 text-left transition-colors hover:border-axis">
              <span className="text-[17px] font-semibold">Just look around, without internet</span>
              <span className="mt-1.5 block text-[14px] leading-snug text-ink-2">Uses a tiny bundled sample, so there is nothing to download. Good for a first look.</span>
            </button>
          </div>
        </>
      )}

      {step === "working" && (
        <div className="mt-8">
          <p className="m-0 text-[17px] text-ink-2">{usedOffline ? "Setting everything up…" : "Getting everything ready. This takes a minute or two."}</p>
          <ol className="m-0 mt-6 list-none space-y-4 p-0">
            {PHASES.map((p, i) => {
              const at = PHASES.findIndex((x) => x.id === phase);
              const state = error && i === at ? "failed" : i < at ? "done" : i === at ? "active" : "waiting";
              return (
                <li key={p.id} className={clsx("flex items-center gap-3 text-[16px]", state === "waiting" && "text-muted")}>
                  {state === "done" && <Check size={18} className="text-good-ink" aria-label="Done" />}
                  {state === "active" && <Loader2 size={18} className="animate-spin text-accent" aria-label="In progress" />}
                  {state === "failed" && <TriangleAlert size={18} className="text-critical-ink" aria-label="Problem" />}
                  {state === "waiting" && <span className="inline-block h-[18px] w-[18px] rounded-full border border-axis" aria-label="Waiting" />}
                  {p.label}
                </li>
              );
            })}
          </ol>
          {job && <div className="mt-6"><JobBar job={job} onCancel={cancel} /></div>}
          {error && (
            <div role="alert" className="mt-6 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-4">
              <p className="m-0 font-semibold">{error.title}</p>
              <p className="mt-1 text-sm text-ink-2">{error.body}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button variant="primary" onClick={() => void quickStart(usedOffline ? "sampler" : "tinystories-quick")}>Try again</Button>
                {!usedOffline && <Button onClick={() => void quickStart("sampler")}>Use the offline sample instead</Button>}
                <Button variant="ghost" onClick={() => setStep("choose")}>Back</Button>
              </div>
            </div>
          )}
        </div>
      )}

      {step !== "working" && <button onClick={() => navigate("/setup", { replace: true })} className="mt-12 self-start text-[13.5px] text-muted underline underline-offset-2 hover:text-ink">Skip this and go to set up</button>}
    </div>
  );
}
