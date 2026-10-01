import clsx from "clsx";
import { MessageSquarePlus, Send, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { backend } from "../backend";
import type { ChatEvent, ChatMessage, ChatMode, ChatSessionInfo } from "../bindings";
import { Button } from "../components/Button";
import { describeError } from "../content/errors";
import { fmtCount } from "../lib/format";
import { useLive } from "../state/live";
import { useRuns } from "../state/queries";

const MODES: { id: ChatMode; label: string; help: string; examples: string[] }[] = [
  {
    id: "continue",
    label: "Continue my text",
    help: "You type the start of something; it carries on writing.",
    examples: ["Once upon a time, there was a little boy named Tom. One day he ", "add 4917 + 388 = "],
  },
  {
    id: "conversation",
    label: "Conversation",
    help: "You ask something; it answers. Best for models trained on chat text.",
    examples: ["What are you?", "Tell me a short story."],
  },
];

interface Streaming {
  text: string;
  rows: number[];
  experts: number[][];
}

function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={clsx("relative inline-flex h-6 w-10 shrink-0 items-center rounded-full transition-colors", checked ? "bg-accent" : "bg-axis")}
    >
      <span className={clsx("inline-block h-[18px] w-[18px] rounded-full bg-white shadow transition-transform", checked ? "translate-x-[18px]" : "translate-x-[3px]")} />
    </button>
  );
}

/** Each character shaded by how many thinking steps it took: darker means more thought. */
function Shaded({ text, rows }: { text: string; rows: number[] }) {
  const chars = Array.from(text);
  const max = Math.max(1, ...rows);
  return (
    <>
      {chars.map((c, i) => (
        <span key={i} title={rows[i] ? `${rows[i]} thinking step${rows[i] === 1 ? "" : "s"}` : undefined} style={{ background: rows[i] ? `color-mix(in srgb, var(--accent) ${Math.round((rows[i]! / max) * 38)}%, transparent)` : undefined }}>
          {c}
        </span>
      ))}
    </>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <p className="text-[13px] text-muted">Appears once it has written something.</p>;
  const max = Math.max(...values, 1);
  const w = 260;
  const h = 56;
  const pts = values.slice(-80).map((v, i, a) => `${(i / Math.max(1, a.length - 1)) * w},${h - (v / max) * (h - 6) - 3}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Thinking steps for each recent character">
      <polyline points={pts} fill="none" stroke="var(--lane-7)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function Chat() {
  const runs = useRuns();
  const [params] = useSearchParams();
  const trainingActive = useLive((s) => s.runState === "running" || s.runState === "preparing");
  const choices = useMemo(() => (runs.data ?? []).filter((r) => r.charsRead > 0 && r.status !== "created"), [runs.data]);

  const [runId, setRunId] = useState<number | null>(null);
  const [mode, setMode] = useState<ChatMode>("continue");
  const [session, setSession] = useState<ChatSessionInfo | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState<Streaming | null>(null);
  const [input, setInput] = useState("");
  const [shade, setShade] = useState(false);
  const [error, setError] = useState<{ title: string; body: string } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const buffer = useRef<Streaming>({ text: "", rows: [], experts: [] });
  const raf = useRef<number | null>(null);

  // Preselect the model from the link (?run=3), otherwise the most recent run that has progress.
  useEffect(() => {
    if (runId != null || choices.length === 0) return;
    const wanted = Number(params.get("run"));
    setRunId(choices.find((c) => c.id === wanted)?.id ?? choices[0]!.id);
  }, [choices, params, runId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming?.text]);

  const refresh = useCallback(async (id: number) => setMessages(await backend.chatGetMessages(id)), []);

  const open = async () => {
    if (runId == null) return;
    setError(null);
    setNotice(null);
    try {
      const s = await backend.chatOpen({ runId, checkpointId: null, mode });
      setSession(s);
      setMessages([]);
    } catch (e) {
      setError(describeError(e));
    }
  };

  const onEvent = useCallback(
    (sid: number) => (e: ChatEvent) => {
      if (e.type === "chars") {
        buffer.current = { text: buffer.current.text + e.text, rows: [...buffer.current.rows, ...e.rows], experts: [...buffer.current.experts, e.experts] };
        // Characters can arrive faster than the screen refreshes; paint at most once per frame.
        if (raf.current == null) {
          raf.current = requestAnimationFrame(() => {
            raf.current = null;
            setStreaming({ ...buffer.current });
          });
        }
      } else if (e.type === "done") {
        setStreaming(null);
        setBusy(false);
        void refresh(sid);
      } else if (e.type === "learned") {
        // The message itself shows what it learned; just reload it.
        void refresh(sid);
      } else if (e.type === "error") {
        setStreaming(null);
        setBusy(false);
        setError(describeError(e.error));
      }
    },
    [refresh],
  );

  const send = async (text = input) => {
    if (!session || !text.trim() || busy) return;
    setError(null);
    setNotice(null);
    setBusy(true);
    setInput("");
    buffer.current = { text: "", rows: [], experts: [] };
    setStreaming({ text: "", rows: [], experts: [] });
    setMessages((m) => [...m, { id: -1, role: "user", content: text, learned: false, learnNatsBefore: null, learnNatsAfter: null, rows: [], charsPerSec: null, createdAt: Date.now() }]);
    try {
      await backend.chatSend(session.id, text, { maxNew: 320, adapt: true }, onEvent(session.id));
    } catch (e) {
      setBusy(false);
      setStreaming(null);
      setError(describeError(e));
      void refresh(session.id);
    }
  };

  const newChat = async () => {
    if (session) await backend.chatClose(session.id).catch(() => {});
    setSession(null);
    setMessages([]);
    setStreaming(null);
    setNotice(null);
    setBusy(false);
  };

  const toggleLearn = async (v: boolean) => {
    if (!session) return;
    try {
      setSession(await backend.chatSetLearn(session.id, v));
    } catch (e) {
      setError(describeError(e));
    }
  };

  const keepLearned = async () => {
    if (!session) return;
    try {
      await backend.chatSaveAdapted(session.id);
      setSession({ ...session, hasAdaptedCopy: true });
      setNotice("Saved as a separate copy. Your original run is unchanged.");
    } catch (e) {
      setError(describeError(e));
    }
  };

  const lastModel = [...messages].reverse().find((m) => m.role === "model");
  const rowsForPanel = streaming ? streaming.rows : (lastModel?.rows ?? []);
  const avgRows = rowsForPanel.length ? rowsForPanel.reduce((a, b) => a + b, 0) / rowsForPanel.length : null;
  const recentExperts = streaming ? Array.from(new Set(streaming.experts.slice(-12).flat())).slice(0, 12) : [];
  const anyLearned = messages.some((m) => m.learned);
  const modeInfo = MODES.find((m) => m.id === (session?.mode ?? mode))!;

  // ── before a chat starts ────────────────────────────────────────────────────────────────────────────────────────
  if (!session) {
    return (
      <div className="mx-auto max-w-[760px] px-8 py-10">
        <h1 className="m-0 text-[26px] font-semibold tracking-tight">Chat with your model</h1>
        <p className="mt-2 max-w-[60ch] text-[15px] leading-relaxed text-ink-2">
          Try out what it has learned. It writes one character at a time, and you can watch how hard it thinks about each one.
        </p>

        {choices.length === 0 ? (
          <p className="mt-10 text-[15px] text-ink-2">
            There is nothing to chat with yet. Start a training and let it save its progress, which happens every few minutes.
          </p>
        ) : (
          <>
            <section className="mt-9">
              <label className="block">
                <span className="text-lg font-semibold">Which model?</span>
                <select
                  value={runId ?? ""}
                  onChange={(e) => setRunId(Number(e.target.value))}
                  className="mt-2 block h-10 w-full max-w-[460px] rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-[15px]"
                >
                  {choices.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name} ({fmtCount(r.charsRead)} characters read)
                    </option>
                  ))}
                </select>
              </label>
              {trainingActive && <p className="mt-2 max-w-[56ch] text-[13px] text-ink-2">A training is running, so replies may be slower while it shares your computer.</p>}
            </section>

            <section className="mt-9" aria-labelledby="mode-h">
              <h2 id="mode-h" className="m-0 text-lg font-semibold">How should it respond?</h2>
              <div role="radiogroup" aria-labelledby="mode-h" className="mt-3 grid gap-3 sm:grid-cols-2">
                {MODES.map((m) => (
                  <label key={m.id} className={clsx("cursor-pointer rounded-[var(--radius-panel)] border bg-surface p-4", mode === m.id ? "border-accent ring-1 ring-accent" : "border-hairline hover:border-axis")}>
                    <input type="radio" name="mode" className="sr-only" checked={mode === m.id} onChange={() => setMode(m.id)} />
                    <span className="block font-semibold">{m.label}</span>
                    <span className="mt-1 block text-[13.5px] text-ink-2">{m.help}</span>
                  </label>
                ))}
              </div>
            </section>

            {error && (
              <div role="alert" className="mt-6 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-3">
                <p className="m-0 font-semibold">{error.title}</p>
                <p className="mt-1 text-sm text-ink-2">{error.body}</p>
              </div>
            )}
            <Button className="mt-8" variant="primary" size="lg" onClick={open} disabled={runId == null}>
              Start chatting
            </Button>
          </>
        )}
      </div>
    );
  }

  // ── an open chat ────────────────────────────────────────────────────────────────────────────────────────────────
  return (
    <div className="mx-auto flex h-full max-w-[1180px] flex-col px-8 py-6">
      <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h1 className="m-0 text-[22px] font-semibold tracking-tight">{session.title}</h1>
        <span className="text-sm text-ink-2">{session.modelLabel}</span>
        <span className="text-sm text-ink-2">{modeInfo.label}</span>
        <Button className="ml-auto" size="sm" icon={<MessageSquarePlus size={15} />} onClick={newChat}>New chat</Button>
      </header>

      <div className="mt-5 grid min-h-0 flex-1 gap-x-12 gap-y-6 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="flex min-h-0 flex-col">
          <div className="min-h-[300px] flex-1 overflow-y-auto pr-2" aria-live="polite">
            {messages.length === 0 && !streaming && (
              <div className="py-10">
                <p className="m-0 text-[15px] font-medium">{session.mode === "continue" ? "Type the start of something." : "Ask it something."}</p>
                <p className="mt-1 text-[13.5px] text-ink-2">Or try one of these:</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {modeInfo.examples.map((ex) => (
                    <button key={ex} onClick={() => setInput(ex)} className="rounded-[var(--radius-control)] border border-hairline bg-surface px-3 py-1.5 text-left text-[13.5px] text-ink-2 hover:border-axis hover:text-ink">
                      {ex.trim()}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <ol className="m-0 list-none space-y-6 p-0">
              {messages.map((m, i) => (
                <li key={m.id === -1 ? `pending-${i}` : m.id} className={clsx(m.role === "user" ? "flex justify-end" : "")}>
                  {m.role === "user" ? (
                    <p className="model-text m-0 max-w-[80%] rounded-[var(--radius-panel)] bg-surface-2 px-4 py-2.5 text-[15px]">{m.content}</p>
                  ) : (
                    <div>
                      <p className="model-text m-0 font-serif text-[19px] leading-[1.6]">{shade ? <Shaded text={m.content} rows={m.rows} /> : m.content}</p>
                      {m.learned && m.learnNatsBefore != null && m.learnNatsAfter != null && (
                        <p className="mt-1.5 text-[12.5px] text-good-ink">Learned from this exchange ({(m.learnNatsBefore / Math.LN2).toFixed(2)} → {(m.learnNatsAfter / Math.LN2).toFixed(2)} bits on that text).</p>
                      )}
                    </div>
                  )}
                </li>
              ))}
              {streaming && (
                <li>
                  <p className="model-text m-0 font-serif text-[19px] leading-[1.6]">
                    {streaming.text || <span className="text-muted">Thinking…</span>}
                    <span aria-hidden className="pulse-dot ml-0.5 inline-block h-4 w-[2px] translate-y-[3px] bg-ink" />
                  </p>
                </li>
              )}
            </ol>
            <div ref={bottom} />
          </div>

          {error && (
            <div role="alert" className="mt-3 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-2.5">
              <p className="m-0 text-sm"><span className="font-semibold">{error.title}.</span> <span className="text-ink-2">{error.body}</span></p>
            </div>
          )}
          {notice && <p role="status" className="mt-3 text-[13px] text-good-ink">{notice}</p>}

          <form
            className="mt-4 flex items-end gap-2 border-t border-hairline pt-4"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <label className="flex-1">
              <span className="sr-only">Your message</span>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send();
                  }
                }}
                rows={2}
                placeholder={session.mode === "continue" ? "Start a story, a sum, a sentence…" : "Ask a question…"}
                className="block w-full resize-none rounded-[var(--radius-control)] border border-hairline bg-surface px-3 py-2.5 text-[15px]"
              />
            </label>
            {busy ? (
              <Button type="button" icon={<Square size={14} />} onClick={() => void backend.chatStop(session.id)}>Stop</Button>
            ) : (
              <Button type="submit" variant="primary" icon={<Send size={15} />} disabled={!input.trim()}>Send</Button>
            )}
          </form>
        </div>

        <aside className="space-y-7 lg:border-l lg:border-hairline lg:pl-8" aria-label="How the model is thinking">
          <section>
            <h2 className="m-0 text-[15px] font-semibold">How it thinks</h2>
            <p className="mt-1 text-[13px] text-ink-2">
              {avgRows != null ? <>On average <span className="font-semibold text-ink">{avgRows.toFixed(1)}</span> thinking steps per character.</> : "Thinking steps per character appear here."}
            </p>
            <div className="mt-2"><Sparkline values={rowsForPanel} /></div>
            <label className="mt-3 flex cursor-pointer items-center gap-2 text-[13px] text-ink-2">
              <input type="checkbox" checked={shade} onChange={(e) => setShade(e.target.checked)} className="accent-[var(--accent)]" />
              Shade each character by how hard it thought
            </label>
            {recentExperts.length > 0 && (
              <p className="mt-3 text-[13px] text-ink-2">
                Experts just used:{" "}
                <span className="font-medium text-ink">{recentExperts.join(", ")}</span>
              </p>
            )}
          </section>

          {session.supportsLearn && (
            <section>
              <div className="flex items-center justify-between gap-3">
                <h2 className="m-0 text-[15px] font-semibold">Learn from this chat</h2>
                <Switch checked={session.learnEnabled} onChange={toggleLearn} label="Learn from this chat" />
              </div>
              <p className="mt-2 text-[13px] leading-snug text-ink-2">
                It keeps learning from what you write together. This can make it slightly worse at other things. What it learns goes into a separate copy; your original run is never changed.
              </p>
              {(anyLearned || session.hasAdaptedCopy) && (
                <Button className="mt-3" size="sm" onClick={keepLearned} disabled={session.hasAdaptedCopy && !anyLearned}>
                  {session.hasAdaptedCopy ? "Saved. Save again" : "Keep what it learned"}
                </Button>
              )}
            </section>
          )}

          <section>
            <Button size="sm" variant="ghost" onClick={() => session && backend.chatReset(session.id).then(() => refresh(session.id))}>Clear this conversation</Button>
          </section>
        </aside>
      </div>
    </div>
  );
}
