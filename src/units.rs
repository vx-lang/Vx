//===- units.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Byte-quantity units, and exact conversion to the one baseline representation: **bytes**.
//
// Every capacity, granule and bandwidth in a machine file passes through here, so this is the
// single place a declared figure becomes a number the cost model uses. It exists because the
// previous inline conversion lost 9% and nobody could see it: `TB` was `2^40` while every `spec:`
// citation in `fleet/` quotes a vendor figure in decimal (NVIDIA's "3.35 TB/s" is 3.35e12 B/s, from
// a 5120-bit bus at 3.2 Gbps). The compiler therefore read every link as ~10% faster than its own
// citation claimed, and understated every predicted transfer time by 9.05% -- an error that in a
// calibration study gets attributed to *hardware* rather than to a prefix convention.
//
// Two rules follow, and both are enforced rather than documented:
//
//   1. SI prefixes are decimal (`GB` = 10^9), IEC prefixes are binary (`GiB` = 2^30). This is the
//      standard reading, so a figure copied from a spec sheet means what the spec sheet meant.
//      Both spellings are accepted, because both occur in the wild: memory *bandwidth* is quoted
//      decimal, cache and shared-memory *capacity* binary.
//   2. Conversion is exact integer arithmetic, never float. `(value * mult as f64).round()` is
//      accurate for the fleet's current figures but silently is not in general, and "silently" is
//      the part that matters -- a declaration that cannot be represented as a whole number of bytes
//      is rejected, not rounded.
//
//===----------------------------------------------------------------------===//

/// A byte-quantity unit suffix. Baseline is [`ByteUnit::B`]; [`ByteUnit::factor`] converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteUnit {
    B,
    // SI (decimal) -- what vendors quote bandwidth in.
    KB,
    MB,
    GB,
    TB,
    PB,
    // IEC (binary) -- what cache/SMEM capacities actually are.
    KiB,
    MiB,
    GiB,
    TiB,
    PiB,
}

impl ByteUnit {
    /// Bytes per unit. `u64` holds every one of these exactly (`PiB` = 2^50, `PB` = 10^15).
    pub const fn factor(self) -> u64 {
        match self {
            ByteUnit::B => 1,
            ByteUnit::KB => 1_000,
            ByteUnit::MB => 1_000_000,
            ByteUnit::GB => 1_000_000_000,
            ByteUnit::TB => 1_000_000_000_000,
            ByteUnit::PB => 1_000_000_000_000_000,
            ByteUnit::KiB => 1 << 10,
            ByteUnit::MiB => 1 << 20,
            ByteUnit::GiB => 1 << 30,
            ByteUnit::TiB => 1 << 40,
            ByteUnit::PiB => 1 << 50,
        }
    }

    /// Parse a unit suffix. Case-sensitive on purpose: `Mb` (megabit) is not `MB` (megabyte), and
    /// accepting it as one would be exactly the class of silent factor-of-8 error this module
    /// exists to prevent.
    pub fn parse(s: &str) -> Option<ByteUnit> {
        Some(match s {
            "B" => ByteUnit::B,
            "KB" => ByteUnit::KB,
            "MB" => ByteUnit::MB,
            "GB" => ByteUnit::GB,
            "TB" => ByteUnit::TB,
            "PB" => ByteUnit::PB,
            "KiB" => ByteUnit::KiB,
            "MiB" => ByteUnit::MiB,
            "GiB" => ByteUnit::GiB,
            "TiB" => ByteUnit::TiB,
            "PiB" => ByteUnit::PiB,
            _ => return None,
        })
    }

    /// Every spelling this module accepts, for diagnostics.
    pub const ALL: &'static str = "B/KB/MB/GB/TB/PB or KiB/MiB/GiB/TiB/PiB";
}

/// Why a declared quantity could not become a whole number of bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitError {
    /// The mantissa was not a decimal number (`3.3.5`, `1e9`, `-4`).
    Malformed,
    /// Exact, but larger than `u64` can hold.
    Overflow,
    /// Exact conversion is not a whole number of bytes, e.g. `1.1 KiB` = 1126.4 B. Rounding here
    /// would be a silent 0.04% error in a model whose whole subject is prediction accuracy.
    NotWholeBytes,
}

/// Convert a decimal mantissa plus a unit into bytes, **exactly**.
///
/// The mantissa arrives as the source text (`"3.35"`) rather than an `f64` so no precision is lost
/// before this function is reached: 3.35 is not representable in binary floating point, and while
/// `3.35 * 1e12` happens to round to the right integer, `parse` -> multiply -> `round` is a
/// pipeline whose correctness depends on the specific values, which is not a property anyone can
/// check by reading it.
///
/// Computed as `(digits * factor) / 10^places` in `u128`, and rejected unless the division is
/// exact. `u128` because `PiB` (2^50) times a 15-digit mantissa overflows `u64` mid-computation
/// while the answer still fits.
pub fn to_bytes(mantissa: &str, unit: ByteUnit) -> Result<u64, UnitError> {
    scale_exact(mantissa, unit.factor() as u128)
}

/// A clock frequency unit. Always decimal — nobody has ever meant 2^30 by "GHz".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreqUnit {
    Hz,
    KHz,
    MHz,
    GHz,
}

impl FreqUnit {
    fn factor(self) -> u128 {
        match self {
            FreqUnit::Hz => 1,
            FreqUnit::KHz => 1_000,
            FreqUnit::MHz => 1_000_000,
            FreqUnit::GHz => 1_000_000_000,
        }
    }

    /// Case-sensitive, like `ByteUnit::parse`. `mhz` is not `MHz`.
    pub fn parse(s: &str) -> Option<FreqUnit> {
        match s {
            "Hz" => Some(FreqUnit::Hz),
            "kHz" => Some(FreqUnit::KHz),
            "MHz" => Some(FreqUnit::MHz),
            "GHz" => Some(FreqUnit::GHz),
            _ => None,
        }
    }

    /// Every spelling this module accepts, for diagnostics.
    pub const ALL: &'static str = "Hz/kHz/MHz/GHz";
}

/// Convert a decimal mantissa plus a frequency unit into hertz, **exactly**.
///
/// Same discipline as [`to_bytes`]: the mantissa arrives as source text so nothing is lost to
/// binary floating point before it gets here. `1.98 GHz` is exactly 1_980_000_000 Hz, and a figure
/// that is not a whole number of hertz is refused rather than rounded.
pub fn to_hertz(mantissa: &str, unit: FreqUnit) -> Result<u64, UnitError> {
    scale_exact(mantissa, unit.factor())
}

/// The shared exact-scaling core: `(digits * factor) / 10^places` in `u128`, exact division
/// required.
fn scale_exact(mantissa: &str, factor: u128) -> Result<u64, UnitError> {
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(UnitError::Malformed);
    }
    if !int_part.bytes().all(|c| c.is_ascii_digit())
        || !frac_part.bytes().all(|c| c.is_ascii_digit())
    {
        return Err(UnitError::Malformed);
    }
    // `3.35` -> digits 335, places 2. Leading zeros are harmless; `u128` holds 38 digits, and a
    // mantissa longer than that is malformed for a hardware figure regardless.
    let digits: u128 = format!("{int_part}{frac_part}")
        .parse()
        .map_err(|_| UnitError::Malformed)?;
    let places = u32::try_from(frac_part.len()).map_err(|_| UnitError::Malformed)?;
    let scale = 10u128.checked_pow(places).ok_or(UnitError::Malformed)?;

    let numerator = digits.checked_mul(factor).ok_or(UnitError::Overflow)?;
    if numerator % scale != 0 {
        return Err(UnitError::NotWholeBytes);
    }
    u64::try_from(numerator / scale).map_err(|_| UnitError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clock figures get the same exactness guarantee as byte figures, and for the same reason:
    /// `1.98 GHz` is what converts a `B/cyc` bandwidth into wall time, so a rounding error here
    /// would land directly in every cycle-denominated prediction.
    #[test]
    fn frequencies_convert_exactly_or_are_refused() {
        assert_eq!(to_hertz("1.98", FreqUnit::GHz), Ok(1_980_000_000));
        assert_eq!(to_hertz("1965", FreqUnit::MHz), Ok(1_965_000_000));
        assert_eq!(to_hertz("2.1", FreqUnit::GHz), Ok(2_100_000_000));
        assert_eq!(to_hertz("1", FreqUnit::Hz), Ok(1));
        // Sub-hertz precision is not a whole number of hertz, so it is refused rather than rounded.
        assert_eq!(
            to_hertz("1.0000000001", FreqUnit::Hz),
            Err(UnitError::NotWholeBytes)
        );
        assert_eq!(to_hertz("1e9", FreqUnit::Hz), Err(UnitError::Malformed));
        // Case-sensitive, like byte units: `ghz` is not `GHz`.
        assert_eq!(FreqUnit::parse("GHz"), Some(FreqUnit::GHz));
        assert_eq!(FreqUnit::parse("kHz"), Some(FreqUnit::KHz));
        assert_eq!(FreqUnit::parse("ghz"), None);
        assert_eq!(FreqUnit::parse("KHz"), None);
    }

    /// The bug this module was written for: the fleet's figures must convert to what their `spec:`
    /// citations mean, not to a binary reading of a decimal number.
    #[test]
    fn si_prefixes_are_decimal_and_iec_are_binary() {
        assert_eq!(to_bytes("3.35", ByteUnit::TB), Ok(3_350_000_000_000));
        assert_eq!(to_bytes("2039", ByteUnit::GB), Ok(2_039_000_000_000));
        assert_eq!(to_bytes("80", ByteUnit::GiB), Ok(85_899_345_920));
        assert_eq!(to_bytes("228", ByteUnit::KiB), Ok(233_472));
        // The 9% that was being lost: the same text under the two readings.
        assert_ne!(
            to_bytes("3.35", ByteUnit::TB),
            to_bytes("3.35", ByteUnit::TiB)
        );
    }

    /// Fractional mantissas are exact, or rejected -- never rounded. `7.7 TB/s` is the contested
    /// `b200.vx` figure the calibration study exists to settle, so it had better be 7.7e12 and not
    /// 7.699999...e12.
    #[test]
    fn fractional_mantissas_are_exact_or_refused() {
        assert_eq!(to_bytes("7.7", ByteUnit::TB), Ok(7_700_000_000_000));
        assert_eq!(to_bytes("5.3", ByteUnit::TB), Ok(5_300_000_000_000));
        assert_eq!(to_bytes("1.5", ByteUnit::KiB), Ok(1536));
        // 1.1 KiB is 1126.4 bytes. Rounding it would be a silent error in the one number the
        // model's accuracy is measured against.
        assert_eq!(
            to_bytes("1.1", ByteUnit::KiB),
            Err(UnitError::NotWholeBytes)
        );
    }

    #[test]
    fn malformed_and_overflowing_input_is_refused() {
        assert_eq!(to_bytes("3.3.5", ByteUnit::TB), Err(UnitError::Malformed));
        assert_eq!(to_bytes("1e9", ByteUnit::B), Err(UnitError::Malformed));
        assert_eq!(to_bytes("-4", ByteUnit::B), Err(UnitError::Malformed));
        assert_eq!(to_bytes("", ByteUnit::B), Err(UnitError::Malformed));
        assert_eq!(
            to_bytes("99999999", ByteUnit::PiB),
            Err(UnitError::Overflow)
        );
    }

    /// `Mb` must not parse as `MB`: a megabit read as a megabyte is a factor-of-8 error, and
    /// interconnect figures are routinely quoted in bits.
    #[test]
    fn unit_spellings_are_case_sensitive() {
        assert_eq!(ByteUnit::parse("MB"), Some(ByteUnit::MB));
        assert_eq!(ByteUnit::parse("MiB"), Some(ByteUnit::MiB));
        assert_eq!(ByteUnit::parse("Mb"), None);
        assert_eq!(ByteUnit::parse("mb"), None);
        assert_eq!(ByteUnit::parse("KIB"), None);
    }

    /// Every unit's factor is what its name says, checked against literals rather than against the
    /// same expression that computes it.
    #[test]
    fn factors_match_their_names() {
        assert_eq!(ByteUnit::KB.factor(), 1_000);
        assert_eq!(ByteUnit::PB.factor(), 1_000_000_000_000_000);
        assert_eq!(ByteUnit::KiB.factor(), 1024);
        assert_eq!(ByteUnit::GiB.factor(), 1_073_741_824);
        assert_eq!(ByteUnit::PiB.factor(), 1_125_899_906_842_624);
    }
}
