//! Publishing a finished file (spec B.11): every writer writes a hidden temporary file beside the
//! destination, validates it, and only then atomically renames it over the destination, so readers
//! only ever see a complete, valid file (the old one or the new one).

use std::fs;
use std::path::Path;

pub use crate::container::temp_path_for;
use crate::validate::{validate, Report};

#[derive(Debug)]
pub enum PublishError {
    Io(std::io::Error),
    /// The file just written failed validation; the destination was left untouched.
    Invalid {
        dst: String,
        report: Box<Report>,
    },
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::Io(e) => write!(f, "{e}"),
            PublishError::Invalid { dst, report } => {
                let issues: Vec<String> = report.errors().iter().take(5).map(|i| i.to_string()).collect();
                let issues = if issues.is_empty() { "fatal structural error".to_string() } else { issues.join("; ") };
                write!(f, "{dst} was not written: the result failed validation: {issues}")
            }
        }
    }
}

impl std::error::Error for PublishError {}

/// Validate `tmp` (unless `check` is false), then atomically replace `dst` with it. On a
/// validation failure `tmp` is deleted and `dst` is untouched.
pub fn publish(tmp: &Path, dst: &Path, check: bool) -> Result<Option<Report>, PublishError> {
    let mut rep = None;
    if check {
        let mut r = validate(tmp);
        r.path = dst.display().to_string(); // the file the report is about, once published
        if !r.ok() {
            let _ = fs::remove_file(tmp);
            return Err(PublishError::Invalid { dst: dst.display().to_string(), report: Box::new(r) });
        }
        rep = Some(r);
    }
    fs::rename(tmp, dst).map_err(PublishError::Io)?;
    Ok(rep)
}
