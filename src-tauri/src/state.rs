//! Application state shared by every command.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use minagi_store::Store;
use minagi_types::{Ctl, EngineFactory, HardwareInfo};

use crate::chat::ChatHost;
use crate::datasets::JobRegistry;
use crate::hub::LiveHub;

/// Where everything the app stores lives. Paths in the database are relative to `root`, so the folder is relocatable.
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub root: PathBuf,
}

impl AppPaths {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn db(&self) -> PathBuf {
        self.root.join("minagi.db")
    }

    pub fn datasets(&self) -> PathBuf {
        self.root.join("datasets")
    }

    /// Turn a path stored relative to the data directory into an absolute one.
    pub fn resolve(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::create_dir_all(self.datasets())
    }
}

/// Flags the recorder reads to tell a user-requested stop from a goal being reached.
#[derive(Default)]
pub struct SessionFlags {
    pub user_stop: AtomicBool,
}

/// The one active training session (at most one runs at a time).
pub struct SessionHandle {
    pub run_id: i64,
    /// Stops the computer sleeping while the session lives; released when the handle is dropped.
    pub _power: crate::power::PowerGuard,
    pub ctl: flume::Sender<Ctl>,
    pub flags: Arc<SessionFlags>,
}

/// Shows a desktop notification: `(title, body)`. Set once at startup; absent in tests.
pub type Notifier = Arc<dyn Fn(&str, &str) + Send + Sync>;

#[derive(Clone)]
pub struct AppState {
    pub paths: Arc<AppPaths>,
    pub store: Store,
    pub factory: Arc<dyn EngineFactory>,
    pub live: LiveHub,
    pub chat: ChatHost,
    pub jobs: JobRegistry,
    pub session: Arc<Mutex<Option<SessionHandle>>>,
    pub hardware: Arc<OnceLock<HardwareInfo>>,
    pub notifier: Arc<OnceLock<Notifier>>,
}

impl AppState {
    pub fn new(paths: AppPaths, store: Store, factory: Arc<dyn EngineFactory>) -> Self {
        Self {
            paths: Arc::new(paths),
            store,
            factory,
            live: LiveHub::default(),
            chat: ChatHost::default(),
            jobs: JobRegistry::default(),
            session: Arc::new(Mutex::new(None)),
            hardware: Arc::new(OnceLock::new()),
            notifier: Arc::new(OnceLock::new()),
        }
    }

    /// Hardware description, probed once and cached.
    pub fn hardware(&self) -> &HardwareInfo {
        self.hardware.get_or_init(|| crate::hardware::probe(self.factory.as_ref(), &self.paths.root))
    }

    pub fn data_dir(&self) -> &Path {
        &self.paths.root
    }
}
