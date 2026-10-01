//! Describe this computer: CPU, memory, free disk, and which compute backends the engine can use.

use std::path::Path;

use minagi_types::{BackendKind, EngineFactory, HardwareInfo};
use sysinfo::{Disks, System};

const GB: f64 = 1_073_741_824.0;

pub fn probe(factory: &dyn EngineFactory, data_dir: &Path) -> HardwareInfo {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let backends = factory.probe_backends();
    // Prefer a GPU backend when one is usable.
    let selected = [BackendKind::Cuda, BackendKind::Metal]
        .into_iter()
        .find(|k| backends.iter().any(|b| b.kind == *k && b.available))
        .unwrap_or(BackendKind::Cpu);

    HardwareInfo {
        os: format!(
            "{} {}",
            System::name().unwrap_or_else(|| std::env::consts::OS.to_string()),
            System::os_version().unwrap_or_default()
        )
        .trim()
        .to_string(),
        cpu: sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Unknown CPU".into()),
        cores: System::physical_core_count().unwrap_or_else(|| sys.cpus().len()).max(1) as u32,
        ram_gb: sys.total_memory() as f64 / GB,
        ram_free_gb: sys.available_memory() as f64 / GB,
        disk_free_gb: free_disk_gb(data_dir),
        backends,
        selected,
    }
}

/// Free space on the disk that holds `path` (the disk with the longest matching mount point).
pub fn free_disk_gb(path: &Path) -> f64 {
    let target = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let disks = Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| target.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space() as f64 / GB)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minagi_mock::{MockFactory, Scenario};

    #[test]
    fn probe_reports_sane_values() {
        let f = MockFactory::new(1.0, Scenario::Normal);
        let hw = probe(&f, &std::env::temp_dir());
        assert!(hw.ram_gb > 0.5, "ram {}", hw.ram_gb);
        assert!(hw.cores >= 1);
        assert!(hw.disk_free_gb > 0.0, "disk {}", hw.disk_free_gb);
        assert_eq!(hw.selected, BackendKind::Metal, "the mock offers a simulated GPU");
        assert!(!hw.cpu.is_empty() && !hw.os.is_empty());
    }

    #[test]
    fn no_gpu_selects_cpu() {
        let f = MockFactory::new(1.0, Scenario::NoGpu);
        assert!(probe(&f, &std::env::temp_dir()).is_cpu_only());
    }
}
