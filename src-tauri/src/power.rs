//! Keep the computer awake while a training runs, so closing the lid timer or an idle screensaver does not stall it.
//! Only idle sleep is prevented; the display is still allowed to turn off.

/// Holds the "stay awake" request for as long as it lives.
pub struct PowerGuard {
    _awake: Option<keepawake::KeepAwake>,
}

impl PowerGuard {
    /// Ask the OS not to sleep. If the platform refuses, training still works: the guard is simply inert.
    pub fn acquire() -> Self {
        let awake = keepawake::Builder::default()
            .display(false)
            .idle(true)
            .sleep(true)
            .reason("Training a language model")
            .app_name("LLM Trainer")
            .app_reverse_domain("com.example.llmtrainer")
            .create()
            .inspect_err(|e| eprintln!("[power] could not keep the computer awake: {e}"))
            .ok();
        Self { _awake: awake }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquiring_and_dropping_never_panics() {
        // CI runners may have no power manager; either way the guard must be safe to hold and drop.
        drop(PowerGuard::acquire());
    }
}
