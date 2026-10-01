import clsx from "clsx";
import { BarChart, CustomChart, LineChart } from "echarts/charts";
import { AxisPointerComponent, DataZoomComponent, GridComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams, LineSeriesOption, SeriesOption } from "echarts";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { RotateCcw, Table2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ChartId } from "../content/charts";
import { fmtCount } from "../lib/format";
import { usePrefs } from "../state/prefs";
import { InfoPopover } from "./explain";

echarts.use([LineChart, BarChart, CustomChart, GridComponent, TooltipComponent, DataZoomComponent, MarkLineComponent, AxisPointerComponent, CanvasRenderer]);

/** Canvas cannot use CSS variables, so tokens like `var(--lane-1)` are resolved to their current value. */
export function resolveColor(token: string): string {
  const m = /^var\((--[\w-]+)\)$/.exec(token);
  if (!m) return token;
  return getComputedStyle(document.documentElement).getPropertyValue(m[1]!).trim() || "#888";
}

export interface ChartLine {
  key: string;
  name: string;
  /** A CSS variable token such as `var(--lane-1)`. */
  color: string;
  x: number[];
  y: (number | null)[];
  /** Optional min/max envelope drawn as a faint band behind the line. */
  lo?: (number | null)[];
  hi?: (number | null)[];
  dashed?: boolean;
  /** Draw each point (for sparse series such as evaluations). */
  points?: boolean;
  faint?: boolean;
}

export interface Annotation {
  x: number;
  kind: "checkpoint" | "jump";
  label: string;
}

interface Props {
  title: string;
  chart: ChartId;
  lines: ChartLine[];
  /** y-axis description shown under the title, e.g. "bits per character, lower is better". */
  caption?: string;
  yFormat: (v: number) => string;
  xFormat?: (v: number) => string;
  /** Charts sharing a group share one crosshair. */
  group: string;
  height?: number;
  annotations?: Annotation[];
  refLine?: { y: number; label: string };
  loading?: boolean;
  empty?: string;
  yLog?: boolean;
  yMin?: number;
  yMax?: number;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, css?: string, text?: string) => {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text !== undefined) e.textContent = text; // always textContent: series and lane names are untrusted
  return e;
};

export function TimeSeriesChart({
  title,
  chart,
  lines,
  caption,
  yFormat,
  xFormat = (v) => `${fmtCount(v)} characters read`,
  group,
  height = 220,
  annotations = [],
  refLine,
  loading,
  empty = "Nothing to chart yet. It appears as the model reads.",
  yLog,
  yMin,
  yMax,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [table, setTable] = useState(false);
  const themeVersion = usePrefs((s) => s.themeVersion);

  const visible = useMemo(() => lines.filter((l) => !hidden.has(l.key)), [lines, hidden]);
  const hasData = lines.some((l) => l.x.length > 0);

  // Create the chart once and keep it sized to its box.
  useEffect(() => {
    if (!host.current) return;
    const c = echarts.init(host.current, undefined, { renderer: "canvas" });
    c.group = group;
    echarts.connect(group);
    instance.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(host.current);
    return () => {
      ro.disconnect();
      c.dispose();
      instance.current = null;
    };
  }, [group]);

  useEffect(() => {
    const c = instance.current;
    if (!c) return;
    const ink2 = resolveColor("var(--ink-2)");
    const muted = resolveColor("var(--muted)");
    const hairline = resolveColor("var(--hairline)");
    const axis = resolveColor("var(--axis)");
    const surface = resolveColor("var(--surface)");
    const ink = resolveColor("var(--ink)");
    const font = getComputedStyle(document.body).fontFamily;

    const series: SeriesOption[] = [];
    visible.forEach((l, idx) => {
      const color = resolveColor(l.color);
      if (l.lo && l.hi) {
        const data = l.x.map((x, i) => [x, l.lo![i] ?? null, l.hi![i] ?? null]);
        series.push({
          id: `band:${l.key}`,
          type: "custom",
          silent: true,
          tooltip: { show: false },
          z: 1,
          data,
          renderItem: (params: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI) => {
            const i = params.dataIndex;
            if (i >= data.length - 1) return null;
            const a = data[i]!, b = data[i + 1]!;
            if (a[1] == null || a[2] == null || b[1] == null || b[2] == null) return null;
            const p = [api.coord([a[0]!, a[2]]), api.coord([b[0]!, b[2]]), api.coord([b[0]!, b[1]]), api.coord([a[0]!, a[1]])];
            return { type: "polygon", shape: { points: p as number[][] }, style: { fill: color, opacity: 0.14 } };
          },
        } as SeriesOption);
      }
      const main: LineSeriesOption = {
        id: `line:${l.key}`,
        name: l.name,
        type: "line",
        z: 3,
        data: l.x.map((x, i) => [x, l.y[i] ?? null]),
        showSymbol: !!l.points,
        symbol: "circle",
        symbolSize: l.points ? 7 : 8,
        connectNulls: false,
        lineStyle: { width: 2, color, type: l.dashed ? "dashed" : "solid", opacity: l.faint ? 0.55 : 1 },
        itemStyle: { color, borderColor: surface, borderWidth: 2 },
        emphasis: { disabled: false, scale: false, itemStyle: { color, borderColor: surface, borderWidth: 2 } },
        sampling: undefined,
      };
      if (idx === 0 && (annotations.length || refLine)) {
        main.markLine = {
          silent: true,
          symbol: "none",
          animation: false,
          label: { show: false },
          data: [
            ...annotations.map((a) => ({ xAxis: a.x, lineStyle: { type: "dashed" as const, width: 1, color: a.kind === "jump" ? resolveColor("var(--accent)") : axis } })),
            ...(refLine ? [{ yAxis: refLine.y, lineStyle: { type: "dashed" as const, width: 1, color: muted }, label: { show: true, formatter: refLine.label, position: "insideEndTop" as const, color: muted, fontFamily: font, fontSize: 11 } }] : []),
          ],
        };
      }
      series.push(main);
    });

    c.setOption(
      {
        animation: false,
        textStyle: { fontFamily: font },
        grid: { left: 56, right: 14, top: 10, bottom: 26 },
        axisPointer: { link: [{ xAxisIndex: "all" }] },
        tooltip: {
          trigger: "axis",
          confine: true,
          transitionDuration: 0,
          backgroundColor: surface,
          borderColor: hairline,
          borderWidth: 1,
          padding: 0,
          extraCssText: "box-shadow:0 8px 30px rgb(0 0 0 / 0.14);border-radius:10px;",
          axisPointer: { type: "line", snap: true, lineStyle: { color: axis, width: 1 } },
          formatter: (raw: unknown) => {
            const params = (Array.isArray(raw) ? raw : [raw]) as { seriesId?: string; seriesName?: string; value?: (number | null)[]; axisValue?: number }[];
            const rows = params.filter((p) => String(p.seriesId ?? "").startsWith("line:"));
            if (!rows.length) return "";
            const x = rows[0]!.value?.[0] ?? params[0]?.axisValue ?? 0;
            const box = el("div", `padding:10px 12px;font:13px ${font};min-width:170px;`);
            box.appendChild(el("div", `color:${ink2};margin-bottom:6px;`, xFormat(x)));
            for (const p of rows) {
              const line = visible.find((l) => `line:${l.key}` === p.seriesId);
              const v = p.value?.[1];
              if (v == null) continue;
              const row = el("div", "display:flex;align-items:center;gap:8px;margin-top:3px;");
              row.appendChild(el("span", `display:inline-block;width:14px;height:2px;border-radius:1px;background:${resolveColor(line?.color ?? "var(--ink)")};`));
              row.appendChild(el("strong", `color:${ink};font-weight:650;`, yFormat(v)));
              row.appendChild(el("span", `color:${ink2};`, p.seriesName ?? ""));
              box.appendChild(row);
            }
            return box;
          },
        },
        xAxis: {
          type: "value",
          min: "dataMin",
          max: "dataMax",
          axisLabel: { color: muted, fontSize: 11, hideOverlap: true, formatter: (v: number) => fmtCount(v) },
          axisLine: { lineStyle: { color: axis } },
          axisTick: { show: false },
          splitLine: { show: false },
        },
        yAxis: {
          type: yLog ? "log" : "value",
          scale: true,
          min: yMin,
          max: yMax,
          axisLabel: { color: muted, fontSize: 11, formatter: (v: number) => yFormat(v) },
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: hairline, width: 1 } },
        },
        dataZoom: [{ type: "inside", xAxisIndex: 0, filterMode: "none", zoomOnMouseWheel: "ctrl", moveOnMouseMove: true, moveOnMouseWheel: false }],
        series,
      },
      { replaceMerge: ["series"] },
    );
  }, [visible, annotations, refLine, yFormat, xFormat, yLog, yMin, yMax, themeVersion]);

  const toggle = (key: string) =>
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(key)) n.delete(key);
      else n.add(key);
      return n;
    });

  return (
    <section aria-label={title} className="min-w-0">
      <header className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <h3 className="m-0 text-[15px] font-semibold">{title}</h3>
        <InfoPopover chart={chart} />
        {caption && <span className="text-[13px] text-muted">{caption}</span>}
        <div className="ml-auto flex items-center gap-1">
          {lines.length >= 2 && (
            <ul className="m-0 flex list-none flex-wrap items-center gap-1 p-0">
              {lines.map((l) => (
                <li key={l.key}>
                  <button
                    onClick={() => toggle(l.key)}
                    aria-pressed={!hidden.has(l.key)}
                    className={clsx("inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[13px] transition-opacity hover:bg-surface-2", hidden.has(l.key) ? "opacity-45" : "opacity-100")}
                  >
                    <span className="inline-block h-[2px] w-[14px] rounded-sm" style={{ background: l.color, opacity: l.faint ? 0.6 : 1 }} />
                    <span className="text-ink-2">{l.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button aria-label="Reset zoom" title="Reset zoom" onClick={() => instance.current?.dispatchAction({ type: "dataZoom", start: 0, end: 100 })} className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-ink">
            <RotateCcw size={14} />
          </button>
          <button aria-label="Show as table" aria-pressed={table} title="Show as table" onClick={() => setTable((t) => !t)} className={clsx("rounded-md p-1.5 hover:bg-surface-2", table ? "text-accent" : "text-muted hover:text-ink")}>
            <Table2 size={14} />
          </button>
        </div>
      </header>

      <div className={clsx("relative", loading && "refetching")} style={{ height }}>
        <div ref={host} className={clsx("absolute inset-0", (table || !hasData) && "invisible")} role="img" aria-label={`${title} chart`} />
        {!hasData && <p className="absolute inset-0 flex items-center justify-center px-6 text-center text-[13px] text-muted">{empty}</p>}
        {table && hasData && <DataTable lines={visible} yFormat={yFormat} xFormat={(v) => fmtCount(v)} />}
      </div>

      {annotations.length > 0 && (
        <p className="mt-1 text-xs text-muted">
          {annotations.some((a) => a.kind === "jump") && <span className="mr-3">Blue dashes: learning rate raised.</span>}
          {annotations.some((a) => a.kind === "checkpoint") && <span>Grey dashes: progress saved.</span>}
        </p>
      )}
    </section>
  );
}

/** Every value the tooltip shows, reachable without hovering. */
function DataTable({ lines, yFormat, xFormat }: { lines: ChartLine[]; yFormat: (v: number) => string; xFormat: (v: number) => string }) {
  const densest = lines.reduce((a, b) => (b.x.length > a.x.length ? b : a), lines[0]!);
  const step = Math.max(1, Math.ceil(densest.x.length / 40));
  const xs = densest.x.filter((_, i) => i % step === 0 || i === densest.x.length - 1);
  const valueAt = (l: ChartLine, x: number) => {
    let best = -1;
    for (let i = 0; i < l.x.length && l.x[i]! <= x; i++) best = i;
    return best < 0 ? null : l.y[best] ?? null;
  };
  return (
    <div className="absolute inset-0 overflow-auto rounded-md border border-hairline bg-surface">
      <table className="w-full border-collapse text-[13px]">
        <thead className="sticky top-0 bg-surface">
          <tr>
            <th scope="col" className="px-3 py-1.5 text-left font-medium text-ink-2">Characters read</th>
            {lines.map((l) => (
              <th key={l.key} scope="col" className="px-3 py-1.5 text-right font-medium text-ink-2">{l.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {xs.map((x) => (
            <tr key={x} className="border-t border-hairline">
              <th scope="row" className="px-3 py-1 text-left font-normal">{xFormat(x)}</th>
              {lines.map((l) => {
                const v = valueAt(l, x);
                return <td key={l.key} className="px-3 py-1 text-right">{v == null ? "–" : yFormat(v)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
