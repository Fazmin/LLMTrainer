import clsx from "clsx";
import { Info } from "lucide-react";
import { Popover } from "radix-ui";
import type { ReactNode } from "react";
import { CHART_INFO } from "../content/charts";
import { GLOSSARY, type TermKey } from "../content/glossary";
import { SECTION_INFO } from "../content/help";

const INFO = { ...CHART_INFO, ...SECTION_INFO };
export type InfoTopic = keyof typeof INFO;

function Bubble({ children }: { children: ReactNode }) {
  return (
    <Popover.Portal>
      <Popover.Content
        side="bottom"
        align="start"
        sideOffset={6}
        collisionPadding={12}
        className="z-50 max-w-[300px] rounded-[var(--radius-panel)] border border-hairline bg-surface p-3.5 text-[13px] leading-relaxed text-ink-2 shadow-[0_8px_30px_rgb(0_0_0/0.12)]"
      >
        {children}
        <Popover.Arrow className="fill-surface" />
      </Popover.Content>
    </Popover.Portal>
  );
}

/** The small "what is this?" button next to a chart title or section heading. */
export function InfoPopover({ topic }: { topic: InfoTopic }) {
  const info = INFO[topic];
  return (
    <Popover.Root>
      <Popover.Trigger
        aria-label={`What is “${info.title}”?`}
        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-muted hover:text-ink data-[state=open]:text-accent"
      >
        <Info size={15} strokeWidth={2} />
      </Popover.Trigger>
      <Bubble>
        <p className="font-semibold text-ink">{info.title}</p>
        <p className="mt-1">{info.what}</p>
        <p className="mt-2">{info.how}</p>
      </Bubble>
    </Popover.Root>
  );
}

/**
 * A section heading with its info button beside it. The button sits outside the heading so the heading's accessible
 * name stays just its text.
 */
export function InfoHeading({ topic, level = 2, id, className, children }: { topic: InfoTopic; level?: 2 | 3; id?: string; className?: string; children?: ReactNode }) {
  const H = level === 2 ? "h2" : "h3";
  return (
    <div className={clsx("flex items-center gap-1.5", className)}>
      <H id={id} className={clsx("m-0 font-semibold", level === 2 ? "text-lg" : "text-[15px]")}>{children ?? INFO[topic].title}</H>
      <InfoPopover topic={topic} />
    </div>
  );
}

/** A word with a dotted underline that explains itself on click or focus. Used wherever a hard term appears. */
export function Term({ k, children }: { k: TermKey; children?: ReactNode }) {
  const term = GLOSSARY[k];
  return (
    <Popover.Root>
      <Popover.Trigger className="cursor-help border-b border-dotted border-muted text-inherit hover:border-ink-2">
        {children ?? term.label}
      </Popover.Trigger>
      <Bubble>
        <p className="font-semibold text-ink">{term.label}</p>
        <p className="mt-1">{term.text}</p>
      </Bubble>
    </Popover.Root>
  );
}
