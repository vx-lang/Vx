//! Project configuration, independent of the Vx compiler and its backend.

use serde::Deserialize;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const FORMAT_CONFIG_FILE: &str = "vx-format.toml";
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FormatConfig {
    /// Spaces per indentation level. Zero disables indentation.
    pub indent: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self { indent: 2 }
    }
}

/// Return the built-in formatter defaults as a complete configuration file.
pub fn default_format_config_toml() -> String {
    let defaults = FormatConfig::default();
    format!("[format]\nindent = {}\n", defaults.indent)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FormatProjectConfig {
    format: FormatConfig,
}

#[derive(Debug)]
pub struct ConfigError {
    pub path: PathBuf,
    message: String,
}

impl ConfigError {
    fn new(path: &Path, message: impl fmt::Display) -> Self {
        Self {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

impl std::error::Error for ConfigError {}

/// Parse an existing file's contents. Missing settings are errors.
pub fn parse_format_config(path: &Path, text: &str) -> Result<FormatConfig, ConfigError> {
    toml::from_str::<FormatProjectConfig>(text)
        .map(|project| project.format)
        .map_err(|error| ConfigError::new(path, error))
}

#[derive(Debug)]
pub struct ResolvedFormatConfig {
    pub settings: FormatConfig,
    /// None means automatic discovery found no configuration file.
    pub source: Option<PathBuf>,
}

fn exists(path: &Path) -> Result<bool, ConfigError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(ConfigError::new(path, error)),
    }
}

/// Search the input's real parent directories, including the Git root.
pub fn discover_format_config(input: &Path) -> Result<Option<PathBuf>, ConfigError> {
    let real_input = fs::canonicalize(input).map_err(|error| ConfigError::new(input, error))?;
    for directory in real_input.parent().into_iter().flat_map(Path::ancestors) {
        let candidate = directory.join(FORMAT_CONFIG_FILE);
        if exists(&candidate)? {
            return Ok(Some(candidate));
        }
        if exists(&directory.join(".git"))? {
            break;
        }
    }
    Ok(None)
}

/// Resolve formatter settings before formatting. CLI overrides do not hide file errors.
pub fn resolve_format_config(
    input: &Path,
    explicit_config: Option<&Path>,
    indent_override: Option<usize>,
) -> Result<ResolvedFormatConfig, ConfigError> {
    let source = match explicit_config {
        Some(path) if path.is_absolute() => Some(path.to_path_buf()),
        Some(path) => Some(
            std::env::current_dir()
                .map_err(|error| ConfigError::new(path, error))?
                .join(path),
        ),
        None => discover_format_config(input)?,
    };
    let mut settings = match &source {
        Some(path) => {
            let text = fs::read_to_string(path).map_err(|error| ConfigError::new(path, error))?;
            parse_format_config(path, &text)?
        }
        None => FormatConfig::default(),
    };
    if let Some(indent) = indent_override {
        settings.indent = indent;
    }
    Ok(ResolvedFormatConfig { settings, source })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join(".git")).unwrap();
        let input = directory.path().join("input.vx");
        fs::write(&input, "fn main() {}\n").unwrap();
        (directory, input)
    }

    #[test]
    fn parses_indentation_including_zero() {
        for indent in [0, 2, 4] {
            let config = parse_format_config(
                Path::new(FORMAT_CONFIG_FILE),
                &format!("[format]\nindent = {indent}\n"),
            )
            .unwrap();
            assert_eq!(config.indent, indent);
        }
    }

    #[test]
    fn rejects_missing_invalid_and_unknown_settings() {
        for text in [
            "",
            "[format]\n",
            "[format]\nindent = -1",
            "[format]\nindent = 18446744073709551616",
            "[format]\nindent = 2.5",
            "[format]\nindent = 'four'",
            "[format]\nindent = true",
            "[format]\nindent = 2\nindent = 4",
            "[format]\nindent = 2\nwidth = 80",
            "[format]\nindent = 2\n[lints]",
            "[format",
        ] {
            let error = parse_format_config(Path::new(FORMAT_CONFIG_FILE), text).expect_err(text);
            assert!(error.to_string().contains(FORMAT_CONFIG_FILE));
        }
    }

    #[test]
    fn prints_defaults_that_round_trip() {
        let text = default_format_config_toml();
        assert_eq!(text, "[format]\nindent = 2\n");
        assert_eq!(
            parse_format_config(Path::new(FORMAT_CONFIG_FILE), &text).unwrap(),
            FormatConfig::default()
        );
    }

    #[test]
    fn defaults_only_when_no_formatter_file_exists() {
        let (directory, input) = project();
        fs::write(directory.path().join("vx-lint.toml"), "invalid TOML [").unwrap();
        let resolved = resolve_format_config(&input, None, None).unwrap();
        assert_eq!(resolved.settings.indent, 2);
        assert!(resolved.source.is_none());

        fs::write(directory.path().join(FORMAT_CONFIG_FILE), "").unwrap();
        assert!(resolve_format_config(&input, None, Some(4)).is_err());
    }

    #[test]
    fn nearest_file_wins_and_explicit_overrides_apply() {
        let (directory, _) = project();
        fs::write(
            directory.path().join(FORMAT_CONFIG_FILE),
            "[format]\nindent = 4",
        )
        .unwrap();
        let nested = directory.path().join("src");
        fs::create_dir(&nested).unwrap();
        let input = nested.join("input.vx");
        fs::write(&input, "").unwrap();
        let root = resolve_format_config(&input, None, None).unwrap();
        assert_eq!(root.settings.indent, 4);
        fs::write(nested.join(FORMAT_CONFIG_FILE), "[format]\nindent = 8").unwrap();
        assert_eq!(
            resolve_format_config(&input, None, None)
                .unwrap()
                .settings
                .indent,
            8
        );
        assert_eq!(
            resolve_format_config(&input, None, Some(0))
                .unwrap()
                .settings
                .indent,
            0
        );
        let explicit = directory.path().join(FORMAT_CONFIG_FILE);
        assert_eq!(
            resolve_format_config(&input, Some(&explicit), None)
                .unwrap()
                .settings
                .indent,
            4
        );
        assert!(resolve_format_config(&input, Some(&nested.join("missing.toml")), None).is_err());
    }

    #[test]
    fn git_file_stops_search_after_checking_its_directory() {
        let (directory, _) = project();
        fs::write(
            directory.path().join(FORMAT_CONFIG_FILE),
            "[format]\nindent = 8",
        )
        .unwrap();
        let nested = directory.path().join("worktree");
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join(".git"), "gitdir: elsewhere").unwrap();
        let input = nested.join("input.vx");
        fs::write(&input, "").unwrap();
        assert_eq!(
            resolve_format_config(&input, None, None)
                .unwrap()
                .settings
                .indent,
            2
        );
        fs::write(nested.join(FORMAT_CONFIG_FILE), "[format]\nindent = 4").unwrap();
        assert_eq!(
            resolve_format_config(&input, None, None)
                .unwrap()
                .settings
                .indent,
            4
        );
    }
}
