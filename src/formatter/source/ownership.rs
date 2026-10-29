use super::{overlaps, SourceFileView};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Legacy,
    Syntax,
    Excluded,
    OpaqueMacro,
    OpaqueUnsupported,
}

impl Owner {
    pub fn preserves_bytes(self) -> bool {
        matches!(
            self,
            Self::Excluded | Self::OpaqueMacro | Self::OpaqueUnsupported
        )
    }
}

#[derive(Clone, Debug)]
pub struct OwnedSpan {
    pub bytes: Range<usize>,
    pub owner: Owner,
}

/// Protected children may sit inside syntax-owned structures. Other overlapping claims are errors.
pub struct OwnershipMap {
    length: usize,
    spans: Vec<OwnedSpan>,
}

impl OwnershipMap {
    pub(crate) fn protect_union(&mut self, ranges: &[Range<usize>]) -> Result<(), String> {
        if ranges
            .iter()
            .any(|range| range.start > range.end || range.end > self.length)
        {
            return Err("Invalid preserved source envelope".into());
        }
        self.spans = ranges
            .iter()
            .cloned()
            .map(|bytes| OwnedSpan {
                bytes,
                owner: Owner::OpaqueUnsupported,
            })
            .collect();
        Ok(())
    }

    pub fn from_source(source: &SourceFileView<'_>) -> Self {
        let mut spans: Vec<_> = source
            .exclusions
            .iter()
            .cloned()
            .map(|bytes| OwnedSpan {
                bytes,
                owner: Owner::Excluded,
            })
            .collect();
        spans.extend(source.macros.iter().map(|view| OwnedSpan {
            bytes: view.bytes.clone(),
            owner: Owner::OpaqueMacro,
        }));
        Self {
            length: source.text.len(),
            spans,
        }
    }

    pub fn claim(&mut self, bytes: Range<usize>, owner: Owner) -> Result<(), String> {
        if bytes.start >= bytes.end || bytes.end > self.length {
            return Err("Invalid source ownership range".into());
        }
        for previous in &self.spans {
            if !overlaps(&bytes, &previous.bytes) {
                continue;
            }
            let contains = bytes.start <= previous.bytes.start && bytes.end >= previous.bytes.end;
            if owner == Owner::Syntax && previous.owner.preserves_bytes() && contains {
                continue;
            }
            return Err("Overlapping source ownership claims".into());
        }
        self.spans.push(OwnedSpan { bytes, owner });
        Ok(())
    }

    pub fn owner_at(&self, byte: usize) -> Owner {
        self.spans
            .iter()
            .filter(|span| span.bytes.contains(&byte))
            .min_by_key(|span| (!span.owner.preserves_bytes(), span.bytes.len()))
            .map_or(Owner::Legacy, |span| span.owner)
    }

    pub fn validate_legacy_edit(&self, bytes: Range<usize>) -> Result<(), String> {
        if bytes.end > self.length || bytes.start > bytes.end {
            return Err("Invalid legacy edit range".into());
        }
        if self.spans.iter().any(|span| {
            overlaps(&span.bytes, &bytes)
                || (bytes.is_empty()
                    && span.bytes.start < bytes.start
                    && bytes.start < span.bytes.end)
        }) {
            return Err("Legacy edit crosses a protected ownership boundary".into());
        }
        Ok(())
    }

    pub fn protected_spans(&self) -> impl Iterator<Item = &OwnedSpan> {
        self.spans
            .iter()
            .filter(|span| span.owner.preserves_bytes())
    }
}

#[derive(Clone, Debug)]
pub struct OutputSegment {
    pub output: Range<usize>,
    pub source: Option<Range<usize>>,
    pub owner: Owner,
    pub whitespace_only: bool,
    pub preserved: bool,
}

impl OutputSegment {
    pub fn allows_legacy_cleanup(&self, map: &OwnershipMap, source: &SourceFileView<'_>) -> bool {
        self.owner == Owner::Legacy
            && self.whitespace_only
            && self.source.as_ref().is_some_and(|bytes| {
                map.validate_legacy_edit(bytes.clone()).is_ok()
                    && !source.gaps.iter().any(|gap| {
                        let start = source.tokens[gap.left.0].bytes.end;
                        let end = source.tokens[gap.right.0].bytes.start;
                        bytes.start <= end && start <= bytes.end
                    })
            })
    }
}
