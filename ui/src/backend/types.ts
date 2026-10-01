import type {
  AppInfo,
  ArithmeticParams,
  DatasetDetail,
  DatasetSummary,
  ExportRequest,
  ExportResult,
  ImportPreview,
  JobEvent,
  LaneMode,
  PathProbe,
  SplitConfig,
  StarterInfo,
  StorageUsage,
  TextPreview,
  ChatEvent,
  ChatMessage,
  ChatOpenRequest,
  ChatParams,
  ChatSessionInfo,
  CheckpointInfo,
  CheckpointMeta,
  CreateRunRequest,
  ForkRunRequest,
  Estimate,
  EvalPoint,
  HardwareInfo,
  LiveMsg,
  LiveSnapshot,
  ModelConfig,
  PoolSnapshot,
  PresetConfig,
  Recommendation,
  RunEvent,
  RunSummary,
  SampleRound,
  SeriesData,
  SeriesRequest,
  StepAt,
  TrainConfig,
} from "../bindings";

/** Everything the UI asks of the backend. Implemented by the Tauri app and by an in-browser fixture. */
export interface Backend {
  readonly kind: "tauri" | "fixture";
  appInfo(): Promise<AppInfo>;
  hardwareInfo(): Promise<HardwareInfo>;
  presetConfigs(): Promise<PresetConfig[]>;
  recommendPreset(datasetId: number | null): Promise<Recommendation>;
  estimateRun(model: ModelConfig, train: TrainConfig, datasetId: number | null): Promise<Estimate>;

  createRun(req: CreateRunRequest): Promise<RunSummary>;
  startRun(runId: number): Promise<RunSummary>;
  /** Continue a run's saved model as a new run, with different settings or text. */
  forkRun(req: ForkRunRequest): Promise<RunSummary>;
  pauseRun(): Promise<void>;
  resumeRun(): Promise<void>;
  stopRun(save: boolean): Promise<void>;
  checkpointNow(): Promise<void>;
  sampleNow(): Promise<void>;
  evalNow(): Promise<void>;
  listRuns(): Promise<RunSummary[]>;
  getRun(runId: number): Promise<RunSummary>;
  getRunConfig(runId: number): Promise<PresetConfig>;
  getActiveRun(): Promise<RunSummary | null>;
  renameRun(runId: number, name: string, notes: string): Promise<RunSummary>;
  deleteRun(runId: number, deleteFiles: boolean): Promise<void>;

  /** Subscribe to the ordered live feed; resolves with the current state. */
  subscribeLive(onMessage: (msg: LiveMsg) => void): Promise<LiveSnapshot>;

  getSeries(req: SeriesRequest): Promise<SeriesData[]>;
  getEvalPoints(runId: number): Promise<EvalPoint[]>;
  getEvalDomains(runId: number): Promise<string[]>;
  getEvents(runId: number, kinds?: string[]): Promise<RunEvent[]>;
  getSamples(runId: number, step?: number): Promise<SampleRound | null>;
  listSampleSteps(runId: number): Promise<StepAt[]>;
  getPoolSnapshot(runId: number, step?: number): Promise<PoolSnapshot | null>;
  listPoolSteps(runId: number): Promise<StepAt[]>;
  listCheckpoints(runId: number): Promise<CheckpointInfo[]>;

  chatOpen(req: ChatOpenRequest): Promise<ChatSessionInfo>;
  chatResume(sessionId: number): Promise<ChatSessionInfo>;
  /** Sends a message; the reply streams to `onEvent`. Resolves with the stored user message id. */
  chatSend(sessionId: number, text: string, params: ChatParams, onEvent: (e: ChatEvent) => void): Promise<number>;
  chatStop(sessionId: number): Promise<void>;
  chatReset(sessionId: number): Promise<void>;
  chatSetLearn(sessionId: number, enabled: boolean): Promise<ChatSessionInfo>;
  chatSaveAdapted(sessionId: number): Promise<CheckpointMeta>;
  chatClose(sessionId: number): Promise<void>;
  chatListSessions(): Promise<ChatSessionInfo[]>;
  chatGetMessages(sessionId: number): Promise<ChatMessage[]>;
  chatDeleteSession(sessionId: number): Promise<void>;

  // ── text the model reads ──
  listDatasets(): Promise<DatasetSummary[]>;
  getDataset(datasetId: number): Promise<DatasetDetail>;
  listStarters(): Promise<StarterInfo[]>;
  probePaths(paths: string[]): Promise<PathProbe[]>;
  /** Folders or files to build a dataset from. Progress streams to `onEvent`; the first message carries the job id. */
  addFolders(paths: string[], mode: LaneMode, onEvent: (e: JobEvent) => void): Promise<DatasetDetail>;
  rebuildDataset(datasetId: number, split: SplitConfig | null, onEvent: (e: JobEvent) => void): Promise<DatasetDetail>;
  getSplit(datasetId: number): Promise<SplitConfig>;
  updateLane(datasetId: number, lane: string, patch: { enabled?: boolean; displayName?: string; samplePrompt?: string }): Promise<DatasetDetail>;
  previewText(datasetId: number, lane: string | null, nChars: number, seed: number): Promise<TextPreview>;
  installStarter(starterId: string, onEvent: (e: JobEvent) => void): Promise<DatasetDetail>;
  generateArithmetic(params: ArithmeticParams, onEvent: (e: JobEvent) => void): Promise<DatasetDetail>;
  deleteDataset(datasetId: number, deleteFiles: boolean): Promise<void>;
  cancelJob(jobId: string): Promise<boolean>;

  storageUsage(): Promise<StorageUsage>;
  /** Show a file or folder in Finder / Explorer. */
  revealInFolder(path: string): Promise<void>;
  /** Open a web page in the default browser. */
  openLink(url: string): Promise<void>;

  exportModel(req: ExportRequest, onEvent: (e: JobEvent) => void): Promise<ExportResult>;
  exportSafetensors(req: ExportRequest, onEvent: (e: JobEvent) => void): Promise<ExportResult>;
  /** What importing a folder would do (a folder made by Export, or the original program's weights folder). */
  previewImport(folder: string): Promise<ImportPreview>;
  importModel(folder: string, onEvent: (e: JobEvent) => void): Promise<RunSummary>;
  /** Ask the OS for one folder. `null` when the user cancels. */
  pickFolder(title: string): Promise<string | null>;

  /** Ask the OS for folders. `null` when the user cancels. */
  pickFolders(): Promise<string[] | null>;
  /** Be told when files are dragged over the window. Returns an unsubscribe function. */
  watchDrops(handler: (e: { type: "enter" | "leave" | "drop"; paths: string[] }) => void): () => void;
}
