use std::fmt;

/// Errors of the container layer: I/O, or bytes that are not what they claim to be.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Structurally invalid data (truncated, bad magic, inconsistent sizes, bad JSON, ...).
    Format(String),
    /// A value that cannot be stored (e.g. a non-finite number in the metadata).
    Value(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Format(m) | Error::Value(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub(crate) fn format_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Format(msg.into()))
}
