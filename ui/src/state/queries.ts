import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { ModelConfig, SeriesRequest, TrainConfig } from "../bindings";
import { backend } from "../backend";

/** Query hooks over the backend. History views read these; the live feed (see live.ts) pushes invalidations. */

export const useAppInfo = () => useQuery({ queryKey: ["appInfo"], queryFn: () => backend.appInfo(), staleTime: Infinity });
export const useHardware = () => useQuery({ queryKey: ["hardware"], queryFn: () => backend.hardwareInfo(), staleTime: Infinity });
export const usePresets = () => useQuery({ queryKey: ["presets"], queryFn: () => backend.presetConfigs(), staleTime: Infinity });
export const useRecommendation = (datasetId: number | null) =>
  useQuery({ queryKey: ["recommendation", datasetId], queryFn: () => backend.recommendPreset(datasetId), staleTime: 60_000 });

export const useEstimate = (model: ModelConfig | undefined, train: TrainConfig | undefined, datasetId: number | null) =>
  useQuery({
    queryKey: ["estimate", model, train, datasetId],
    queryFn: () => backend.estimateRun(model!, train!, datasetId),
    enabled: !!model && !!train,
    placeholderData: keepPreviousData,
  });

export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: () => backend.listRuns() });
export const useRun = (id: number | null) =>
  useQuery({ queryKey: ["run", id], queryFn: () => backend.getRun(id!), enabled: id != null });

/** Chart series. While a run is live the data is re-read every few seconds; buckets are aligned so nothing jitters. */
export function useSeries(req: SeriesRequest | null, live: boolean) {
  return useQuery({
    queryKey: ["series", req],
    queryFn: () => backend.getSeries(req!),
    enabled: !!req && req.keys.length > 0,
    placeholderData: keepPreviousData,
    refetchInterval: live ? 2500 : false,
    staleTime: live ? 0 : Infinity,
  });
}

export const useEvalDomains = (runId: number | null, version: number) =>
  useQuery({ queryKey: ["evalDomains", runId, version], queryFn: () => backend.getEvalDomains(runId!), enabled: runId != null, placeholderData: keepPreviousData });

export const useEvents = (runId: number | null, version: number) =>
  useQuery({ queryKey: ["events", runId, version], queryFn: () => backend.getEvents(runId!), enabled: runId != null, placeholderData: keepPreviousData });

export const useSampleSteps = (runId: number | null, version: number) =>
  useQuery({ queryKey: ["sampleSteps", runId, version], queryFn: () => backend.listSampleSteps(runId!), enabled: runId != null, placeholderData: keepPreviousData });

/** Sample round at `step`, or the latest one (refetched whenever `version` changes) when `step` is undefined. */
export const useSamples = (runId: number | null, step: number | undefined, version = 0) =>
  useQuery({ queryKey: ["samples", runId, step ?? "latest", step === undefined ? version : 0], queryFn: () => backend.getSamples(runId!, step), enabled: runId != null, placeholderData: keepPreviousData });

export const usePool = (runId: number | null, version: number) =>
  useQuery({ queryKey: ["pool", runId, version], queryFn: () => backend.getPoolSnapshot(runId!), enabled: runId != null, placeholderData: keepPreviousData });

export const useCheckpoints = (runId: number | null, version: number) =>
  useQuery({ queryKey: ["checkpoints", runId, version], queryFn: () => backend.listCheckpoints(runId!), enabled: runId != null, placeholderData: keepPreviousData });

export const useDatasets = () => useQuery({ queryKey: ["datasets"], queryFn: () => backend.listDatasets() });
export const useDataset = (id: number | null) =>
  useQuery({ queryKey: ["dataset", id], queryFn: () => backend.getDataset(id!), enabled: id != null });
export const useStarters = () => useQuery({ queryKey: ["starters"], queryFn: () => backend.listStarters(), staleTime: Infinity });
export const useSplit = (id: number | null) =>
  useQuery({ queryKey: ["split", id], queryFn: () => backend.getSplit(id!), enabled: id != null });
