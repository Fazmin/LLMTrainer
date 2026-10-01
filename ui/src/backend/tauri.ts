import { Channel } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { openUrl, revealItemInDir } from "@tauri-apps/plugin-opener";
import { commands, type ChatEvent, type JobEvent, type LiveMsg } from "../bindings";
import type { Backend } from "./types";

/** The real backend: thin wrappers over the generated, typed Tauri commands. Errors arrive as thrown `AppError`s. */
export const tauriBackend: Backend = {
  kind: "tauri",
  appInfo: () => commands.appInfo(),
  hardwareInfo: () => commands.hardwareInfo(),
  presetConfigs: () => commands.presetConfigs(),
  recommendPreset: (datasetId) => commands.recommendPreset(datasetId),
  estimateRun: (model, train, datasetId) => commands.estimateRun(model, train, datasetId),

  createRun: (req) => commands.createRun(req),
  forkRun: (req) => commands.forkRun(req),
  startRun: (id) => commands.startRun(id),
  pauseRun: async () => void (await commands.pauseRun()),
  resumeRun: async () => void (await commands.resumeRun()),
  stopRun: async (save) => void (await commands.stopRun(save)),
  checkpointNow: async () => void (await commands.checkpointNow()),
  sampleNow: async () => void (await commands.sampleNow()),
  evalNow: async () => void (await commands.evalNow()),
  listRuns: () => commands.listRuns(),
  getRun: (id) => commands.getRun(id),
  getRunConfig: (id) => commands.getRunConfig(id),
  getActiveRun: () => commands.getActiveRun(),
  renameRun: (id, name, notes) => commands.renameRun(id, name, notes),
  deleteRun: async (id, files) => void (await commands.deleteRun(id, files)),

  subscribeLive: (onMessage) => {
    const channel = new Channel<LiveMsg>();
    channel.onmessage = onMessage;
    return commands.subscribeLive(channel);
  },

  getSeries: (req) => commands.getSeries(req),
  getEvalPoints: (id) => commands.getEvalPoints(id),
  getEvalDomains: (id) => commands.getEvalDomains(id),
  getEvents: (id, kinds) => commands.getEvents(id, kinds ?? null),
  getSamples: (id, step) => commands.getSamples(id, step ?? null),
  listSampleSteps: (id) => commands.listSampleSteps(id),
  getPoolSnapshot: (id, step) => commands.getPoolSnapshot(id, step ?? null),
  listPoolSteps: (id) => commands.listPoolSteps(id),
  listCheckpoints: (id) => commands.listCheckpoints(id),

  chatOpen: (req) => commands.chatOpen(req),
  chatResume: (id) => commands.chatResume(id),
  chatSend: (id, text, params, onEvent) => {
    const channel = new Channel<ChatEvent>();
    channel.onmessage = onEvent;
    return commands.chatSend(id, text, params, channel);
  },
  chatStop: async (id) => void (await commands.chatStop(id)),
  chatReset: async (id) => void (await commands.chatReset(id)),
  chatSetLearn: (id, enabled) => commands.chatSetLearn(id, enabled),
  chatSaveAdapted: (id) => commands.chatSaveAdapted(id),
  chatClose: async (id) => void (await commands.chatClose(id)),
  chatListSessions: () => commands.chatListSessions(),
  chatGetMessages: (id) => commands.chatGetMessages(id),
  chatDeleteSession: async (id) => void (await commands.chatDeleteSession(id)),

  listDatasets: () => commands.listDatasets(),
  getDataset: (id) => commands.getDataset(id),
  listStarters: () => commands.listStarters(),
  probePaths: (paths) => commands.probePaths(paths),
  addFolders: (paths, mode, onEvent) => commands.addFolders(paths, mode, jobChannel(onEvent)),
  rebuildDataset: (id, split, onEvent) => commands.rebuildDataset(id, split, jobChannel(onEvent)),
  getSplit: (id) => commands.getSplit(id),
  updateLane: (id, lane, patch) =>
    commands.updateLane(id, lane, patch.enabled ?? null, patch.displayName ?? null, patch.samplePrompt ?? null),
  previewText: (id, lane, nChars, seed) => commands.previewText(id, lane, nChars, seed),
  installStarter: (starterId, onEvent) => commands.installStarter(starterId, jobChannel(onEvent)),
  generateArithmetic: (params, onEvent) => commands.generateArithmetic(params, jobChannel(onEvent)),
  deleteDataset: async (id, files) => void (await commands.deleteDataset(id, files)),
  cancelJob: (jobId) => commands.cancelJob(jobId),

  storageUsage: () => commands.storageUsage(),
  revealInFolder: (path) => revealItemInDir(path),
  openLink: (url) => openUrl(url),

  exportModel: (req, onEvent) => commands.exportModel(req, jobChannel(onEvent)),
  exportSafetensors: (req, onEvent) => commands.exportSafetensors(req, jobChannel(onEvent)),
  previewImport: (folder) => commands.previewImport(folder),
  importModel: (folder, onEvent) => commands.importModel(folder, jobChannel(onEvent)),
  pickFolder: async (title) => {
    const picked = await openDialog({ directory: true, multiple: false, title });
    return typeof picked === "string" ? picked : null;
  },

  pickFolders: async () => {
    const picked = await openDialog({ directory: true, multiple: true, title: "Choose folders of text" });
    if (!picked) return null;
    return Array.isArray(picked) ? picked : [picked];
  },
  watchDrops: (handler) => {
    let off: (() => void) | undefined;
    let closed = false;
    void getCurrentWebview()
      .onDragDropEvent((e) => {
        const p = e.payload;
        if (p.type === "enter") handler({ type: "enter", paths: p.paths });
        else if (p.type === "drop") handler({ type: "drop", paths: p.paths });
        else if (p.type === "leave") handler({ type: "leave", paths: [] });
      })
      .then((un) => {
        if (closed) un();
        else off = un;
      });
    return () => {
      closed = true;
      off?.();
    };
  },
};

function jobChannel(onEvent: (e: JobEvent) => void): Channel<JobEvent> {
  const channel = new Channel<JobEvent>();
  channel.onmessage = onEvent;
  return channel;
}
