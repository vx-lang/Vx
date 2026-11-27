//! Project configuration, independent of the Vx compiler and its backend.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const FORMAT_CONFIG_FILE: &str = "vx-format.toml";

/// A named set of formatter defaults. A new edition can change a default without changing the
/// output for projects that name an older edition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StyleEdition {
    #[default]
    V2026,
}

impl StyleEdition {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "2026" => Some(Self::V2026),
            _ => None,
        }
    }
}

impl fmt::Display for StyleEdition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V2026 => write!(f, "2026"),
        }
    }
}

impl<'de> Deserialize<'de> for StyleEdition {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Accept `"2026"` from a file and `2026` from `--set style_edition=2026`.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Spelling {
            Text(String),
            Number(i64),
        }
        let text = match Spelling::deserialize(deserializer)? {
            Spelling::Text(text) => text,
            Spelling::Number(number) => number.to_string(),
        };
        Self::parse(&text).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "unknown style_edition `{text}`; the known one is 2026"
            ))
        })
    }
}

/// Formatter settings after defaults and overrides are applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatConfig {
    /// The edition whose defaults fill in the settings that the file does not name.
    pub style_edition: StyleEdition,
    /// Spaces per indentation level. Zero disables indentation.
    pub indent: usize,
    /// The preferred maximum width of a line, including indentation.
    pub max_width: usize,
    /// The preferred maximum width of a function header. It is at most `max_width`.
    pub fn_signature_width: usize,
    /// Add a space before single colons in function headers.
    pub space_before_colon: bool,
    /// The widest that the fields of a struct literal may be, for the literal to stay on one line.
    /// When unset, 40% of `max_width`.
    pub struct_lit_width: usize,
    /// The widest that an `if` value may be, for it to stay on one line. When unset, all of
    /// `max_width`, so only the line width limits it.
    pub single_line_if_else_max_width: usize,
    /// Formatting rules switched on or off by name. A rule that is not named uses its default.
    pub rules: BTreeMap<String, bool>,
}

impl FormatConfig {
    /// The default settings of `edition`.
    pub fn defaults(edition: StyleEdition) -> Self {
        match edition {
            StyleEdition::V2026 => Self {
                style_edition: edition,
                indent: 2,
                max_width: 100,
                fn_signature_width: 100,
                space_before_colon: false,
                struct_lit_width: 40,
                single_line_if_else_max_width: 100,
                rules: BTreeMap::new(),
            },
        }
    }

    /// Whether the rule `name` is on. `default` is the rule's own default for this edition.
    pub fn rule_enabled(&self, name: &str, default: bool) -> bool {
        self.rules.get(name).copied().unwrap_or(default)
    }

    /// The same settings, with the rule `name` switched on or off.
    pub fn with_rule(mut self, name: &str, enabled: bool) -> Self {
        self.rules.insert(name.to_string(), enabled);
        self
    }
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self::defaults(StyleEdition::default())
    }
}

/// The `[format]` table as a file writes it. Only `indent` is required.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSettings {
    style_edition: Option<StyleEdition>,
    indent: usize,
    max_width: Option<usize>,
    fn_signature_width: Option<usize>,
    space_before_colon: Option<bool>,
    struct_lit_width: Option<usize>,
    single_line_if_else_max_width: Option<usize>,
    #[serde(default)]
    rules: BTreeMap<String, bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FormatProjectConfig {
    format: FileSettings,
}

impl FileSettings {
    fn resolve(self) -> Result<FormatConfig, String> {
        let defaults = FormatConfig::defaults(self.style_edition.unwrap_or_default());
        let max_width = self.max_width.unwrap_or(defaults.max_width);
        let fn_signature_width = self.fn_signature_width.unwrap_or(max_width);
        if max_width == 0 {
            return Err("max_width must be greater than zero".into());
        }
        if fn_signature_width == 0 {
            return Err("fn_signature_width must be greater than zero".into());
        }
        if fn_signature_width > max_width {
            return Err(format!(
                "fn_signature_width ({fn_signature_width}) must not be greater than max_width ({max_width})"
            ));
        }
        // A narrower width that the file does not set is a share of max_width.
        let share = |default: usize| default * max_width / defaults.max_width;
        let struct_lit_width = self
            .struct_lit_width
            .unwrap_or(share(defaults.struct_lit_width));
        if struct_lit_width > max_width {
            return Err(format!(
                "struct_lit_width ({struct_lit_width}) must not be greater than max_width ({max_width})"
            ));
        }
        let single_line_if_else_max_width = self
            .single_line_if_else_max_width
            .unwrap_or(share(defaults.single_line_if_else_max_width));
        if single_line_if_else_max_width > max_width {
            return Err(format!(
                "single_line_if_else_max_width ({single_line_if_else_max_width}) must not be greater than max_width ({max_width})"
            ));
        }
        Ok(FormatConfig {
            indent: self.indent,
            max_width,
            fn_signature_width,
            space_before_colon: self
                .space_before_colon
                .unwrap_or(defaults.space_before_colon),
            struct_lit_width,
            single_line_if_else_max_width,
            rules: self.rules,
            ..defaults
        })
    }
}

/// The built-in defaults as a complete configuration file. `rules` lists each formatting rule
/// with its default, because only the formatter knows its rules.
pub fn default_format_config_toml(rules: &[(&str, bool)]) -> String {
    let defaults = FormatConfig::default();
    // The narrower widths are left out: unset, each one follows max_width.
    let mut text = format!(
        "[format]\nstyle_edition = \"{}\"\nindent = {}\nmax_width = {}\nspace_before_colon = {}\n\
         # Unset, these widths follow max_width: fn_signature_width is max_width,\n\
         # struct_lit_width is {}% of it, and single_line_if_else_max_width is {}% of it.\n",
        defaults.style_edition,
        defaults.indent,
        defaults.max_width,
        defaults.space_before_colon,
        defaults.struct_lit_width * 100 / defaults.max_width,
        defaults.single_line_if_else_max_width * 100 / defaults.max_width
    );
    if !rules.is_empty() {
        text.push_str("\n[format.rules]\n");
        for (name, enabled) in rules {
            text.push_str(&format!("{name} = {enabled}\n"));
        }
    }
    text
}

/// One `--set KEY=VALUE` from the command line. `KEY` names a setting of the `[format]` table.
/// A dotted key, such as `rules.function_headers`, names a setting in a table inside it.
#[derive(Clone, Debug)]
pub struct SettingOverride {
    text: String,
    key: Vec<String>,
    value: toml::Value,
}

impl SettingOverride {
    pub fn parse(text: &str) -> Result<Self, String> {
        let Some((key, value)) = text.split_once('=') else {
            return Err(format!("`{text}` must have the form KEY=VALUE"));
        };
        let key: Vec<String> = key
            .trim()
            .split('.')
            .map(|part| part.trim().to_string())
            .collect();
        if key.iter().any(String::is_empty) {
            return Err(format!("`{text}` names no setting before `=`"));
        }
        // A TOML value, such as `80` or `true`. Anything else is text, such as `2026`.
        let value = toml::from_str::<toml::Table>(&format!("value = {}", value.trim()))
            .ok()
            .and_then(|mut table| table.remove("value"))
            .unwrap_or_else(|| toml::Value::String(value.trim().to_string()));
        Ok(Self {
            text: text.to_string(),
            key,
            value,
        })
    }

    /// The setting's name, with its parts joined by `.`.
    pub fn key(&self) -> String {
        self.key.join(".")
    }

    /// The text as the command line gave it.
    pub fn text(&self) -> &str {
        &self.text
    }

    fn apply(&self, format: &mut toml::Table) -> Result<(), String> {
        let (last, tables) = self.key.split_last().expect("a parsed key has a part");
        let mut table = format;
        for part in tables {
            table = table
                .entry(part.clone())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                .as_table_mut()
                .ok_or_else(|| format!("`{part}` is a setting, not a table"))?;
        }
        table.insert(last.clone(), self.value.clone());
        Ok(())
    }
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
    parse_with_overrides(text, &[]).map_err(|message| ConfigError::new(path, message))
}

fn parse_with_overrides(text: &str, overrides: &[SettingOverride]) -> Result<FormatConfig, String> {
    let mut table: toml::Table = toml::from_str(text).map_err(|error| error.to_string())?;
    if !overrides.is_empty() {
        let format = table
            .entry("format")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or("`format` must be a table")?;
        for setting in overrides {
            setting.apply(format)?;
        }
    }
    let project: FormatProjectConfig = table
        .try_into()
        .map_err(|error: toml::de::Error| error.to_string())?;
    project.format.resolve()
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

/// Resolve formatter settings before formatting. Overrides do not hide errors in the file.
pub fn resolve_format_config(
    input: &Path,
    explicit_config: Option<&Path>,
    overrides: &[SettingOverride],
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
    let text = match &source {
        Some(path) => {
            let text = fs::read_to_string(path).map_err(|error| ConfigError::new(path, error))?;
            parse_format_config(path, &text)?;
            text
        }
        // Name only the required setting, so that `--set max_width=N` also narrows the
        // signature width, as it would in a file that does not set it.
        None => format!("[format]\nindent = {}\n", FormatConfig::default().indent),
    };
    let settings = parse_with_overrides(&text, overrides).map_err(|message| {
        let given: Vec<_> = overrides
            .iter()
            .map(|setting| setting.text.as_str())
            .collect();
        ConfigError::new(
            Path::new(&format!("--set {}", given.join(" --set "))),
            message,
        )
    })?;
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

    fn parse(text: &str) -> Result<FormatConfig, ConfigError> {
        parse_format_config(Path::new(FORMAT_CONFIG_FILE), text)
    }

    fn set(texts: &[&str]) -> Vec<SettingOverride> {
        texts
            .iter()
            .map(|text| SettingOverride::parse(text).unwrap())
            .collect()
    }

    #[test]
    fn parses_indentation_including_zero() {
        for indent in [0, 2, 4] {
            let config = parse(&format!("[format]\nindent = {indent}\n")).unwrap();
            assert_eq!(config.indent, indent);
            assert_eq!(config.max_width, 100);
            assert_eq!(config.fn_signature_width, 100);
            assert!(!config.space_before_colon);
            assert!(config.rules.is_empty());
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
            "[format]\nindent = 2\nmax_width = 0",
            "[format]\nindent = 2\nfn_signature_width = 0",
            "[format]\nindent = 2\nfn_signature_width = -1",
            "[format]\nindent = 2\nfn_signature_width = 'wide'",
            "[format]\nindent = 2\nfn_signature_width = 18446744073709551616",
            "[format]\nindent = 2\nfn_signature_width = 101",
            "[format]\nindent = 2\nmax_width = 80\nfn_signature_width = 90",
            "[format]\nindent = 2\nstruct_lit_width = 101",
            "[format]\nindent = 2\nsingle_line_if_else_max_width = 101",
            "[format]\nindent = 2\nmax_width = 40\nstruct_lit_width = 50",
            "[format]\nindent = 2\nstruct_lit_width = -1",
            "[format]\nindent = 2\nspace_before_colon = 'true'",
            "[format]\nindent = 2\nspace_before_colon = 1",
            "[format]\nindent = 2\nstyle_edition = '2015'",
            "[format]\nindent = 2\nstyle_edition = true",
            "[format]\nindent = 2\n[format.rules]\nfunction_headers = 'false'",
            "[format]\nindent = 2\n[format.rules]\nfunction_headers = 0",
            "[format]\nindent = 2\nindent = 4",
            "[format]\nindent = 2\nwidth = 80",
            "[format]\nindent = 2\nformat_function_headers = false",
            "[format]\nindent = 2\n[lints]",
            "[format",
        ] {
            let error = parse(text).expect_err(text);
            assert!(error.to_string().contains(FORMAT_CONFIG_FILE));
        }
    }

    #[test]
    fn the_signature_width_follows_max_width_unless_the_file_sets_it() {
        let config = parse("[format]\nindent = 4\nmax_width = 80\n").unwrap();
        assert_eq!((config.max_width, config.fn_signature_width), (80, 80));
        let config =
            parse("[format]\nindent = 4\nmax_width = 80\nfn_signature_width = 60\n").unwrap();
        assert_eq!((config.max_width, config.fn_signature_width), (80, 60));
    }

    #[test]
    fn prints_defaults_that_parse_back_to_the_defaults() {
        let text = default_format_config_toml(&[("function_headers", true)]);
        let config = parse(&text).unwrap();
        assert_eq!(
            config,
            FormatConfig::default().with_rule("function_headers", true)
        );
        assert_eq!(
            parse(&default_format_config_toml(&[])).unwrap(),
            FormatConfig::default()
        );
    }

    #[test]
    fn printed_defaults_still_parse_after_max_width_is_made_smaller() {
        let text = default_format_config_toml(&[]).replace("max_width = 100", "max_width = 40");
        let config = parse(&text).unwrap();
        assert_eq!(config.max_width, 40);
        assert_eq!(config.fn_signature_width, 40);
        assert_eq!(config.struct_lit_width, 16);
        assert_eq!(config.single_line_if_else_max_width, 40);
    }

    #[test]
    fn reads_the_style_edition_and_rules() {
        let config = parse(
            "[format]\nstyle_edition = \"2026\"\nindent = 2\n[format.rules]\nfunction_headers = false\nfuture_rule = true\n",
        )
        .unwrap();
        assert_eq!(config.style_edition, StyleEdition::V2026);
        assert!(!config.rule_enabled("function_headers", true));
        assert!(config.rule_enabled("future_rule", false));
        assert!(config.rule_enabled("unnamed_rule", true));
        assert!(!config.rule_enabled("unnamed_rule", false));
    }

    #[test]
    fn reads_the_struct_literal_width() {
        assert_eq!(
            parse("[format]\nindent = 2\n").unwrap().struct_lit_width,
            40
        );
        let config = parse("[format]\nindent = 2\nstruct_lit_width = 18\n").unwrap();
        assert_eq!(config.struct_lit_width, 18);
        let config = parse("[format]\nindent = 2\nsingle_line_if_else_max_width = 30\n").unwrap();
        assert_eq!(config.single_line_if_else_max_width, 30);
        let config = parse("[format]\nindent = 2\nmax_width = 60\n").unwrap();
        assert_eq!(config.struct_lit_width, 24);
        assert_eq!(config.single_line_if_else_max_width, 60);
    }

    #[test]
    fn reads_both_colon_spacing_styles() {
        for enabled in [true, false] {
            let config = parse(&format!(
                "[format]\nindent = 2\nspace_before_colon = {enabled}\n"
            ))
            .unwrap();
            assert_eq!(config.space_before_colon, enabled);
        }
    }

    #[test]
    fn overrides_go_through_the_same_checks_as_the_file() {
        let (directory, input) = project();
        fs::write(
            directory.path().join(FORMAT_CONFIG_FILE),
            "[format]\nindent = 4\n",
        )
        .unwrap();
        let resolved = resolve_format_config(
            &input,
            None,
            &set(&[
                "max_width=80",
                "space_before_colon = true",
                "rules.function_headers=false",
                "style_edition=2026",
            ]),
        )
        .unwrap();
        assert_eq!(resolved.settings.indent, 4);
        assert_eq!(resolved.settings.max_width, 80);
        assert_eq!(resolved.settings.fn_signature_width, 80);
        assert!(resolved.settings.space_before_colon);
        assert!(!resolved.settings.rule_enabled("function_headers", true));
        for bad in [
            "max_width=0",
            "fn_signature_width=200",
            "width=80",
            "indent=four",
            "style_edition=1999",
            "rules.function_headers=maybe",
            "indent.deeper=1",
        ] {
            let error = resolve_format_config(&input, None, &set(&[bad])).expect_err(bad);
            assert!(error.to_string().contains(bad), "{error}");
        }
        for malformed in ["max_width", "=80", "rules.=true"] {
            assert!(SettingOverride::parse(malformed).is_err(), "{malformed}");
        }
    }

    #[test]
    fn overrides_do_not_hide_errors_in_the_file() {
        let (directory, input) = project();
        fs::write(
            directory.path().join(FORMAT_CONFIG_FILE),
            "[format]\nindent = -1\n",
        )
        .unwrap();
        let error = resolve_format_config(&input, None, &set(&["indent=4"])).unwrap_err();
        assert!(error.to_string().contains(FORMAT_CONFIG_FILE), "{error}");
    }

    #[test]
    fn defaults_only_when_no_formatter_file_exists() {
        let (directory, input) = project();
        fs::write(directory.path().join("vx-lint.toml"), "invalid TOML [").unwrap();
        let resolved = resolve_format_config(&input, None, &[]).unwrap();
        assert_eq!(resolved.settings.indent, 2);
        assert!(resolved.source.is_none());

        fs::write(directory.path().join(FORMAT_CONFIG_FILE), "").unwrap();
        assert!(resolve_format_config(&input, None, &set(&["indent=4"])).is_err());
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
        let root = resolve_format_config(&input, None, &[]).unwrap();
        assert_eq!(root.settings.indent, 4);
        fs::write(nested.join(FORMAT_CONFIG_FILE), "[format]\nindent = 8").unwrap();
        assert_eq!(
            resolve_format_config(&input, None, &[])
                .unwrap()
                .settings
                .indent,
            8
        );
        assert_eq!(
            resolve_format_config(&input, None, &set(&["indent=0"]))
                .unwrap()
                .settings
                .indent,
            0
        );
        let explicit = directory.path().join(FORMAT_CONFIG_FILE);
        assert_eq!(
            resolve_format_config(&input, Some(&explicit), &[])
                .unwrap()
                .settings
                .indent,
            4
        );
        assert!(resolve_format_config(&input, Some(&nested.join("missing.toml")), &[]).is_err());
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
            resolve_format_config(&input, None, &[])
                .unwrap()
                .settings
                .indent,
            2
        );
        fs::write(nested.join(FORMAT_CONFIG_FILE), "[format]\nindent = 4").unwrap();
        assert_eq!(
            resolve_format_config(&input, None, &[])
                .unwrap()
                .settings
                .indent,
            4
        );
    }
}
