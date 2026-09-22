use std::fmt;

use lvf::Report;

/// Everything that can go wrong in fflv. The variant says which stage failed; the message is
/// meant for people.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The container is malformed (see the `lvf` crate).
    Format(String),
    /// A bad layer id, rect, z, ... (bad input, not a bad file).
    Meta(String),
    /// libvpx failed, or produced something that breaks the format's rules.
    Encode(String),
    Decode(String),
    /// FFmpeg / ffprobe failed, or a source file is unusable.
    Media(String),
    Writer(String),
    Edit(String),
    Pack(String),
    /// A render/extract output path or format that cannot be written.
    Output(String),
    View(String),
    /// The file just written failed validation; the destination was left untouched.
    Invalid {
        dst: String,
        report: Box<Report>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Format(m)
            | Error::Meta(m)
            | Error::Encode(m)
            | Error::Decode(m)
            | Error::Media(m)
            | Error::Writer(m)
            | Error::Edit(m)
            | Error::Pack(m)
            | Error::Output(m)
            | Error::View(m) => f.write_str(m),
            Error::Invalid { dst, report } => {
                let issues: Vec<String> = report.errors().iter().take(5).map(|i| i.to_string()).collect();
                let issues = if issues.is_empty() { "fatal structural error".to_string() } else { issues.join("; ") };
                write!(f, "{dst} was not written: the result failed validation: {issues}")
            }
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

impl From<lvf::Error> for Error {
    fn from(e: lvf::Error) -> Self {
        match e {
            lvf::Error::Io(e) => Error::Io(e),
            lvf::Error::Format(m) | lvf::Error::Value(m) => Error::Format(m),
        }
    }
}

impl From<lvf::MetaError> for Error {
    fn from(e: lvf::MetaError) -> Self {
        Error::Meta(e.0)
    }
}

impl From<lvf::PublishError> for Error {
    fn from(e: lvf::PublishError) -> Self {
        match e {
            lvf::PublishError::Io(e) => Error::Io(e),
            lvf::PublishError::Invalid { dst, report } => Error::Invalid { dst, report },
        }
    }
}

impl Error {
    /// The same error with its message moved to another stage (e.g. a media error during `pack`).
    pub fn into_stage(self, stage: fn(String) -> Error) -> Error {
        match self {
            Error::Io(_) | Error::Invalid { .. } => self,
            other => stage(other.to_string()),
        }
    }
}
