//! Which accelerator the engine runs on, and the housekeeping each backend needs.
//!
//! - **Metal** (Apple Silicon, macOS 15 or newer) is the default there. Training runs in f32: the spike measured f16/bf16
//!   at only ~6% faster and f16 produces NaNs without loss scaling, so master weights *and* activations stay f32.
//! - **CUDA** (Windows/Linux, opt-in build) and **CPU** (everywhere, Accelerate on macOS) are the others.
//!
//! Every training step on macOS must run inside [`pool`]: candle's Metal backend allocates autoreleased Objective-C
//! objects (command buffers, encoders) and a thread without a run loop only frees them when a pool drains.

use candle_core::{Device, Result};
use minagi_types::{BackendInfo, BackendKind};

/// Run `f` inside an Objective-C autorelease pool on macOS (a plain call elsewhere).
///
/// Wrap every training step, evaluation chunk and generated character in this, or resident memory climbs until the
/// process is killed (candle issues #2271 and #3755).
#[cfg(target_os = "macos")]
pub fn pool<R>(f: impl FnOnce() -> R) -> R {
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(not(target_os = "macos"))]
pub fn pool<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// macOS major version (e.g. 15 for Sequoia), or `None` off macOS or when it cannot be read.
pub fn macos_major() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sw_vers").arg("-productVersion").output().ok()?;
        let text = String::from_utf8(out.stdout).ok()?;
        text.trim().split('.').next()?.parse().ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Why Metal cannot be used here, or `None` when it can (checked without creating a device).
pub fn metal_unavailable_reason() -> Option<String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Some("Metal needs a Mac with Apple Silicon.".into());
    }
    match macos_major() {
        Some(v) if v >= 15 => None,
        Some(_) => Some("The Apple GPU needs macOS 15 or newer; this Mac is on an older version.".into()),
        None => Some("Could not tell which macOS version this is.".into()),
    }
}

/// Open the device for `kind`. Falls back with a readable error instead of panicking.
pub fn device(kind: BackendKind) -> Result<Device> {
    match kind {
        BackendKind::Cpu => Ok(Device::Cpu),
        BackendKind::Metal => {
            if let Some(why) = metal_unavailable_reason() {
                candle_core::bail!("{why}");
            }
            Device::new_metal(0)
        }
        BackendKind::Cuda => {
            if !cfg!(feature = "cuda") {
                candle_core::bail!("This build does not include NVIDIA GPU support.");
            }
            Device::new_cuda(0)
        }
    }
}

/// What the engine can use on this machine, best first. `ram_gb` sizes the CPU budget.
pub fn probe(ram_gb: f64) -> Vec<BackendInfo> {
    let metal_reason = metal_unavailable_reason();
    let metal_ok = metal_reason.is_none() && Device::new_metal(0).is_ok();
    let metal_reason = match (&metal_reason, metal_ok) {
        (Some(r), _) => Some(r.clone()),
        (None, false) => Some("The Apple GPU could not be opened.".into()),
        _ => None,
    };
    // Leave the OS and the app room: GPU work on a unified-memory Mac competes with everything else.
    let metal_budget = (ram_gb * 0.6).max(2.0);
    let cuda_built = cfg!(feature = "cuda");
    let cuda_ok = cuda_built && Device::new_cuda(0).is_ok();
    vec![
        BackendInfo {
            kind: BackendKind::Metal,
            name: "Apple GPU (Metal)".into(),
            available: metal_ok,
            reason: metal_reason,
            mem_budget_gb: if metal_ok { metal_budget } else { 0.0 },
            bf16_ok: false,
        },
        BackendInfo {
            kind: BackendKind::Cuda,
            name: "NVIDIA GPU (CUDA)".into(),
            available: cuda_ok,
            reason: if cuda_ok {
                None
            } else if cuda_built {
                Some("No NVIDIA GPU was found.".into())
            } else {
                Some("This version of the app was built without NVIDIA GPU support.".into())
            },
            mem_budget_gb: 0.0,
            bf16_ok: cuda_ok,
        },
        BackendInfo {
            kind: BackendKind::Cpu,
            name: "Processor (CPU)".into(),
            available: true,
            reason: None,
            mem_budget_gb: (ram_gb * 0.6).max(1.0),
            bf16_ok: false,
        },
    ]
}

/// Process memory in GiB: `(resident, peak_footprint)`. Zeros where the platform gives no answer.
pub fn process_memory_gb() -> (f64, f64) {
    #[cfg(target_os = "macos")]
    {
        // `struct task_vm_info`: resident_size @16, phys_footprint @144, ledger_phys_footprint_peak @168.
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(task: u32, flavor: u32, info: *mut u32, count: *mut u32) -> i32;
        }
        const TASK_VM_INFO: u32 = 22;
        let mut buf = [0u32; 128];
        let mut count = buf.len() as u32;
        // SAFETY: `buf` outlives the call and `count` tells the kernel its size in 32-bit words.
        let kr = unsafe { task_info(mach_task_self(), TASK_VM_INFO, buf.as_mut_ptr(), &mut count) };
        if kr != 0 {
            return (0.0, 0.0);
        }
        let used = (count as usize * 4).min(buf.len() * 4);
        // SAFETY: reinterpreting initialised u32s as bytes; `used` is within the array.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, used) };
        let rd = |off: usize| {
            bytes.get(off..off + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or([0; 8]))).unwrap_or(0)
        };
        let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        (gb(rd(144)), gb(rd(168).max(rd(144))))
    }
    #[cfg(not(target_os = "macos"))]
    {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_available() {
        let info = probe(16.0);
        let cpu = info.iter().find(|b| b.kind == BackendKind::Cpu).unwrap();
        assert!(cpu.available && cpu.reason.is_none());
        assert!(device(BackendKind::Cpu).is_ok());
    }

    #[test]
    fn unavailable_backends_explain_themselves() {
        for b in probe(16.0) {
            assert_eq!(b.available, b.reason.is_none(), "{:?}", b.kind);
        }
    }

    #[test]
    fn pool_returns_the_closure_value() {
        assert_eq!(pool(|| 41 + 1), 42);
    }
}
