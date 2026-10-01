use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum TextError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("there is no text to read in {0}")]
    NoText(String),
    #[error("every file in the chosen text is shorter than one step ({chunk} characters)")]
    TooShort { chunk: usize },
    #[error("{0}")]
    Invalid(String),
}

pub type TextResult<T> = Result<T, TextError>;

impl TextError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        TextError::Io { path: path.into(), source }
    }
}
