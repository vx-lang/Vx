//===- suggest.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Provides "did you mean?" suggestions for name resolution errors using
// Levenshtein edit distance. Used by E2001-E2007 diagnostics.
//
//===----------------------------------------------------------------------===//

/// Compute Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row = vec![0; b_len + 1];

    for (i, a_char) in a.chars().enumerate() {
        curr_row[0] = i + 1;
        for (j, b_char) in b.chars().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

/// Find the closest match to `target` from a list of `candidates`.
/// Returns `None` if no candidate is within `max_dist` edits.
pub fn closest_match<'a>(target: &str, candidates: &[&'a str], max_dist: usize) -> Option<&'a str> {
    candidates
        .iter()
        .filter_map(|c| {
            let dist = levenshtein(target, c);
            if dist > 0 && dist <= max_dist {
                Some((*c, dist))
            } else {
                None
            }
        })
        .min_by_key(|(_, d)| *d)
        .map(|(c, _)| c)
}

/// Suggest a correction for a name, using a dynamic max distance based on name length.
/// Short names (1-3 chars) allow 1 edit, medium names (4-7) allow 2, longer allow 3.
pub fn suggest_name<'a>(target: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let max_dist = match target.len() {
        0 => return None,
        1..=3 => 1,
        4..=7 => 2,
        _ => 3,
    };
    closest_match(target, candidates, max_dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_levenshtein_one_edit() {
        assert_eq!(levenshtein("count", "conut"), 2); // transposition = 2 edits
        assert_eq!(levenshtein("count", "counr"), 1); // substitution
        assert_eq!(levenshtein("count", "coun"), 1); // deletion
        assert_eq!(levenshtein("count", "counts"), 1); // insertion
    }

    #[test]
    fn test_levenshtein_multiple_edits() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_closest_match_found() {
        let candidates = vec!["count", "counter", "amount", "mount"];
        assert_eq!(closest_match("counr", &candidates, 2), Some("count"));
    }

    #[test]
    fn test_closest_match_too_far() {
        let candidates = vec!["apple", "banana", "cherry"];
        assert_eq!(closest_match("xyz", &candidates, 2), None);
    }

    #[test]
    fn test_closest_match_exact_excluded() {
        // Exact matches (distance 0) should be excluded since the name is already correct
        let candidates = vec!["count", "counter"];
        assert_eq!(closest_match("count", &candidates, 2), Some("counter"));
    }

    #[test]
    fn test_suggest_name_short() {
        let candidates = vec!["x", "y", "z", "a"];
        // 'b' is distance 1 from all candidates, so one will be returned
        assert!(suggest_name("b", &candidates).is_some());
    }

    #[test]
    fn test_suggest_name_medium() {
        let candidates = vec!["counter", "count", "amount"];
        // 'contr' (len 5) allows max_dist=2; 'count' is dist 2 from 'contr'
        let result = suggest_name("contr", &candidates);
        assert!(result == Some("count") || result == Some("counter"));
    }

    #[test]
    fn test_suggest_name_empty() {
        let candidates = vec!["x", "y"];
        assert_eq!(suggest_name("", &candidates), None);
    }
}
