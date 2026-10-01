//! Fan-out of live messages to the UI, plus a snapshot so a reloaded webview can recover immediately.

use std::sync::{Arc, Mutex};

use minagi_types::{LiveMsg, LiveSnapshot, RunSummary};
use tauri::ipc::Channel;

#[derive(Default)]
struct Inner {
    subscribers: Vec<Channel<LiveMsg>>,
    snapshot: LiveSnapshot,
}

#[derive(Clone, Default)]
pub struct LiveHub {
    inner: Arc<Mutex<Inner>>,
}

impl LiveHub {
    /// Register a UI channel and return the current state so it can render without waiting for the next message.
    pub fn subscribe(&self, channel: Channel<LiveMsg>) -> LiveSnapshot {
        let mut g = self.inner.lock().unwrap();
        g.subscribers.push(channel);
        g.snapshot.clone()
    }

    /// Broadcast a message and fold it into the snapshot. Subscribers whose channel is gone are dropped.
    pub fn send(&self, msg: LiveMsg) {
        let mut g = self.inner.lock().unwrap();
        match &msg {
            LiveMsg::Stage { info, .. } => g.snapshot.stage = Some(info.clone()),
            LiveMsg::Pulse { schedule, .. } => g.snapshot.schedule = Some(schedule.clone()),
            LiveMsg::Eval { result, .. } => g.snapshot.last_eval = Some(result.clone()),
            LiveMsg::Pool { snapshot, .. } => g.snapshot.last_pool = Some(snapshot.clone()),
            LiveMsg::Insight { insight, .. } => g.snapshot.insight = Some(insight.clone()),
            _ => {}
        }
        g.subscribers.retain(|ch| ch.send(msg.clone()).is_ok());
    }

    /// Replace the run summary in the snapshot (used when the run row changes).
    pub fn set_run(&self, run: Option<RunSummary>) {
        self.inner.lock().unwrap().snapshot.run = run;
    }

    /// Forget everything about the previous run when a new one starts.
    pub fn reset_snapshot(&self, run: RunSummary) {
        self.inner.lock().unwrap().snapshot = LiveSnapshot { run: Some(run), ..Default::default() };
    }
}
