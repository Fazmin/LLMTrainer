//! Connection management: one writer thread that batches work into transactions, plus a pool of read-only
//! connections. SQLite runs in WAL mode so readers (charts) never block the writer (training telemetry).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use minagi_types::AppError;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;

/// Schema version this build understands (the number of migrations).
pub const SCHEMA_VERSION: u32 = 1;

const BATCH_MAX_MESSAGES: usize = 2000;
const BATCH_MAX_AGE: Duration = Duration::from_millis(500);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("connection pool error: {0}")]
    Pool(#[from] r2d2::Error),
    #[error("migration failed: {0}")]
    Migration(String),
    #[error("this data was written by a newer version of the app (database v{found}, this app supports v{supported})")]
    DatabaseTooNew { found: u32, supported: u32 },
    #[error("the database writer has stopped")]
    Closed,
    #[error("not found: {0}")]
    NotFound(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

impl From<StoreError> for AppError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::NotFound(s) => AppError::NotFound(s),
            StoreError::Io(e) => AppError::Io(e.to_string()),
            other => AppError::Db(other.to_string()),
        }
    }
}

pub type StoreResult<T> = Result<T, StoreError>;

type BatchFn = Box<dyn FnOnce(&Connection) -> rusqlite::Result<()> + Send>;
type SyncFn = Box<dyn FnOnce(&mut Connection) + Send>;

enum Msg {
    /// Fire-and-forget work that may be grouped with other batched work in one transaction.
    Batch(BatchFn),
    /// Work whose result the caller waits for; flushes any open batch first.
    Sync(SyncFn),
    /// Commit everything queued so far, then reply.
    Barrier(flume::Sender<()>),
    Shutdown,
}

struct Inner {
    tx: flume::Sender<Msg>,
    readers: Pool<SqliteConnectionManager>,
    path: PathBuf,
    writer: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
        if let Some(handle) = self.writer.lock().ok().and_then(|mut g| g.take()) {
            let _ = handle.join();
        }
    }
}

/// Handle to the database. Cheap to clone.
#[derive(Clone)]
pub struct Store {
    inner: Arc<Inner>,
}

const PRAGMAS: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -32768;
PRAGMA journal_size_limit = 67108864;
";

/// Ordered schema migrations. `PRAGMA user_version` holds how many have been applied, so adding a migration is
/// appending one entry here (and bumping `SCHEMA_VERSION`).
const MIGRATIONS: &[&str] = &[include_str!("../migrations/0001_init.sql")];

/// Apply every migration the database has not seen yet, each in its own transaction.
fn run_migrations(conn: &mut Connection) -> StoreResult<()> {
    let current = user_version(conn)? as usize;
    for (i, sql) in MIGRATIONS.iter().enumerate().skip(current) {
        let tx = conn.transaction()?;
        tx.execute_batch(sql).map_err(|e| StoreError::Migration(format!("migration {} failed: {e}", i + 1)))?;
        tx.execute_batch(&format!("PRAGMA user_version = {}", i + 1))?;
        tx.commit()?;
    }
    Ok(())
}

fn user_version(conn: &Connection) -> rusqlite::Result<u32> {
    conn.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0)).map(|v| v as u32)
}

impl Store {
    /// Open (creating if needed) the database at `path`, migrating it to the current schema.
    ///
    /// Refuses to touch a database written by a newer app version. Before migrating an existing database a copy is
    /// saved to `backups/` next to it.
    pub fn open(path: &Path) -> StoreResult<Store> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut conn = Connection::open(path)?;
        conn.execute_batch(PRAGMAS)?;

        let found = user_version(&conn)?;
        if found > SCHEMA_VERSION {
            return Err(StoreError::DatabaseTooNew { found, supported: SCHEMA_VERSION });
        }
        if found > 0 && found < SCHEMA_VERSION {
            backup_before_migration(&conn, path, found)?;
        }
        run_migrations(&mut conn)?;

        let (tx, rx) = flume::unbounded::<Msg>();
        let handle = std::thread::Builder::new()
            .name("minagi-db-writer".into())
            .spawn(move || writer_loop(conn, rx))
            .map_err(StoreError::Io)?;

        let manager = SqliteConnectionManager::file(path).with_init(|c| {
            c.execute_batch(
                "PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000; PRAGMA temp_store = MEMORY; \
                 PRAGMA cache_size = -32768; PRAGMA mmap_size = 268435456; PRAGMA query_only = ON;",
            )
        });
        let readers = Pool::builder().max_size(4).build(manager)?;

        Ok(Store { inner: Arc::new(Inner { tx, readers, path: path.to_path_buf(), writer: Mutex::new(Some(handle)) }) })
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Run `f` on the writer thread and wait for its result. Use for commands that need an immediate answer.
    pub fn write<R, F>(&self, f: F) -> StoreResult<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut Connection) -> StoreResult<R> + Send + 'static,
    {
        let (reply_tx, reply_rx) = flume::bounded(1);
        let job: SyncFn = Box::new(move |conn| {
            let _ = reply_tx.send(f(conn));
        });
        self.inner.tx.send(Msg::Sync(job)).map_err(|_| StoreError::Closed)?;
        reply_rx.recv().map_err(|_| StoreError::Closed)?
    }

    /// Queue work to be batched into a transaction with other queued work. Errors are logged, not returned.
    pub fn write_batched<F>(&self, f: F)
    where
        F: FnOnce(&Connection) -> rusqlite::Result<()> + Send + 'static,
    {
        let _ = self.inner.tx.send(Msg::Batch(Box::new(f)));
    }

    /// Block until everything queued so far is committed.
    pub fn flush(&self) {
        let (tx, rx) = flume::bounded(1);
        if self.inner.tx.send(Msg::Barrier(tx)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Run a read-only query on a pooled connection.
    pub fn read<R, F>(&self, f: F) -> StoreResult<R>
    where
        F: FnOnce(&Connection) -> StoreResult<R>,
    {
        let conn = self.inner.readers.get()?;
        f(&conn)
    }
}

fn writer_loop(mut conn: Connection, rx: flume::Receiver<Msg>) {
    let mut in_txn = false;
    let mut count = 0usize;
    let mut started = Instant::now();

    fn commit(conn: &Connection, in_txn: &mut bool) {
        if *in_txn {
            if let Err(e) = conn.execute_batch("COMMIT") {
                eprintln!("[minagi-store] commit failed: {e}");
                let _ = conn.execute_batch("ROLLBACK");
            }
            *in_txn = false;
        }
    }

    loop {
        let msg = if in_txn {
            let remaining = BATCH_MAX_AGE.saturating_sub(started.elapsed());
            match rx.recv_timeout(remaining) {
                Ok(m) => m,
                Err(flume::RecvTimeoutError::Timeout) => {
                    commit(&conn, &mut in_txn);
                    continue;
                }
                Err(flume::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(m) => m,
                Err(_) => break,
            }
        };

        match msg {
            Msg::Batch(f) => {
                if !in_txn {
                    if let Err(e) = conn.execute_batch("BEGIN IMMEDIATE") {
                        eprintln!("[minagi-store] begin failed: {e}");
                        continue;
                    }
                    in_txn = true;
                    started = Instant::now();
                    count = 0;
                }
                if let Err(e) = f(&conn) {
                    eprintln!("[minagi-store] batched write failed: {e}");
                }
                count += 1;
                if count >= BATCH_MAX_MESSAGES || started.elapsed() >= BATCH_MAX_AGE {
                    commit(&conn, &mut in_txn);
                }
            }
            Msg::Sync(f) => {
                commit(&conn, &mut in_txn);
                f(&mut conn);
            }
            Msg::Barrier(reply) => {
                commit(&conn, &mut in_txn);
                let _ = reply.send(());
            }
            Msg::Shutdown => {
                commit(&conn, &mut in_txn);
                break;
            }
        }
    }
    commit(&conn, &mut in_txn);
}

fn backup_before_migration(conn: &Connection, path: &Path, from_version: u32) -> StoreResult<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new(".")).join("backups");
    std::fs::create_dir_all(&dir)?;
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let dest = dir.join(format!("minagi-v{from_version}-{ts}.db"));
    let dest_str = dest.to_string_lossy().replace('\'', "''");
    conn.execute_batch(&format!("VACUUM INTO '{dest_str}'"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_are_valid_and_apply_from_empty() {
        assert_eq!(MIGRATIONS.len() as u32, SCHEMA_VERSION, "SCHEMA_VERSION must match the number of migrations");
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        let n: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM metric_keys", [], |r| r.get(0))?)).unwrap();
        assert_eq!(n, 17);
        let v = store.read(|c| Ok(user_version(c)?)).unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn reopening_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.db");
        drop(Store::open(&p).unwrap());
        let s = Store::open(&p).unwrap();
        assert!(!dir.path().join("backups").exists(), "no backup when already current");
        drop(s);
    }

    #[test]
    fn refuses_a_database_from_a_newer_app() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.db");
        drop(Store::open(&p).unwrap());
        {
            let c = Connection::open(&p).unwrap();
            c.execute_batch(&format!("PRAGMA user_version = {}", SCHEMA_VERSION + 5)).unwrap();
        }
        match Store::open(&p) {
            Err(StoreError::DatabaseTooNew { found, supported }) => {
                assert_eq!(found, SCHEMA_VERSION + 5);
                assert_eq!(supported, SCHEMA_VERSION);
            }
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("expected the open to be refused"),
        }
    }

    #[test]
    fn batched_writes_are_visible_after_flush() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        for i in 0..100 {
            store.write_batched(move |c| {
                c.execute("INSERT INTO app_settings (key, value, updated_at) VALUES (?1, 'v', 0)", [format!("k{i}")])?;
                Ok(())
            });
        }
        store.flush();
        let n: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM app_settings", [], |r| r.get(0))?)).unwrap();
        assert_eq!(n, 100);
    }

    #[test]
    fn sync_write_returns_its_result() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        let id = store
            .write(|c| {
                c.execute("INSERT INTO app_settings (key, value, updated_at) VALUES ('a', 'b', 1)", [])?;
                Ok(c.last_insert_rowid())
            })
            .unwrap();
        assert!(id > 0);
    }
}
