//! Frame/time arithmetic. All rounding is exact integer math, round-half-up (spec A.1). The player
//! (player/src/format/timing.ts) implements the same formulas; they must agree bit for bit because
//! pts_us(f) is what ties decoder output back to its composite frame.

use std::fmt;

/// An exact rational frame rate `num/den` (both positive, reduced).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fps {
    pub num: u32,
    pub den: u32,
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Fps {
    pub fn new(num: u64, den: u64) -> Result<Fps, String> {
        if num == 0 || den == 0 {
            return Err(format!("fps must be positive, got {num}/{den}"));
        }
        let g = gcd(num as u128, den as u128) as u64;
        let (n, d) = (num / g, den / g);
        if n > u32::MAX as u64 || d > u32::MAX as u64 {
            return Err(format!("fps {num}/{den} is out of range"));
        }
        Ok(Fps { num: n as u32, den: d as u32 })
    }

    /// Accepts "30", "30/1", "30000/1001", "29.97".
    pub fn parse(s: &str) -> Result<Fps, String> {
        let s = s.trim();
        if let Some((a, b)) = s.split_once('/') {
            let n: u64 = a.trim().parse().map_err(|_| format!("cannot parse fps {s:?}"))?;
            let d: u64 = b.trim().parse().map_err(|_| format!("cannot parse fps {s:?}"))?;
            return Fps::new(n, d);
        }
        let (n, d) = parse_decimal(s).ok_or_else(|| format!("cannot parse fps {s:?}"))?;
        if n <= 0 {
            return Err(format!("fps must be positive, got {s:?}"));
        }
        Fps::new(n as u64, d as u64)
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl fmt::Display for Fps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// pts_us(f) = round(f * 1_000_000 * den / num), round half up.
pub fn pts_us(frame: u64, fps: Fps) -> i64 {
    let n = fps.num as u128;
    ((2 * frame as u128 * 1_000_000 * fps.den as u128 + n) / (2 * n)) as i64
}

/// A decimal string as an exact fraction (numerator, denominator), e.g. "1.25" → (125, 100).
/// Accepts an optional sign and exponent ("1e3", "2.5E-1").
pub fn parse_decimal(s: &str) -> Option<(i128, i128)> {
    let s = s.trim();
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i32>().ok()?),
        None => (s, 0),
    };
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant.strip_prefix('+').unwrap_or(mant)),
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    if int.is_empty() && frac.is_empty() || !int.chars().chain(frac.chars()).all(|c| c.is_ascii_digit()) {
        return None;
    }
    let digits = format!("{int}{frac}");
    let mut num: i128 = digits.parse().ok()?;
    let mut scale = frac.len() as i32 - exp;
    let mut den: i128 = 1;
    while scale > 0 {
        den = den.checked_mul(10)?;
        scale -= 1;
    }
    while scale < 0 {
        num = num.checked_mul(10)?;
        scale += 1;
    }
    Some((if neg { -num } else { num }, den))
}

/// frame = round(seconds * fps), round half up, with seconds taken at their decimal value
/// (0.1 means exactly 1/10, so `3.0` at 30 fps is exactly frame 90).
pub fn seconds_to_frame(seconds: &str, fps: Fps) -> Result<i64, String> {
    let (n, d) = parse_decimal(seconds).ok_or_else(|| format!("cannot parse seconds {seconds:?}"))?;
    let num = n * fps.num as i128;
    let den = d * fps.den as i128;
    Ok(round_half_up(num, den))
}

/// round(num / den), halves rounded towards +infinity; den > 0.
pub fn round_half_up(num: i128, den: i128) -> i64 {
    (2 * num + den).div_euclid(2 * den) as i64
}

/// Shortest decimal text of an f64 that round-trips (for JSON numbers given as floats).
pub fn f64_decimal(v: f64) -> String {
    let s = format!("{v}");
    if s.contains(['e', 'E', '.']) || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        s + ".0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pts_matches_the_reference_values() {
        let f30 = Fps::new(30, 1).unwrap();
        assert_eq!((0..4).map(|f| pts_us(f, f30)).collect::<Vec<_>>(), vec![0, 33333, 66667, 100000]);
        assert_eq!(pts_us(3600, f30), 120_000_000);
        let ntsc = Fps::new(30000, 1001).unwrap();
        assert_eq!(pts_us(1, ntsc), 33367);
        assert_eq!(pts_us(2, ntsc), 66733);
        assert_eq!(pts_us(30000, ntsc), 1_001_000_000);
    }

    #[test]
    fn pts_rounds_half_up() {
        let f = Fps::new(128, 1).unwrap();
        assert_eq!(pts_us(1, f), 7813);
        assert_eq!(pts_us(3, f), 23438);
    }

    #[test]
    fn round_half_up_on_negatives() {
        assert_eq!(round_half_up(5, 2), 3);
        assert_eq!(round_half_up(7, 2), 4);
        assert_eq!(round_half_up(-1, 2), 0);
    }

    #[test]
    fn fps_parsing() {
        assert_eq!(Fps::parse("30/1").unwrap(), Fps { num: 30, den: 1 });
        assert_eq!(Fps::parse("30000/1001").unwrap(), Fps { num: 30000, den: 1001 });
        assert_eq!(Fps::parse("25").unwrap(), Fps { num: 25, den: 1 });
        assert_eq!(Fps::parse("29.97").unwrap(), Fps { num: 2997, den: 100 });
        assert_eq!(Fps::parse("60/2").unwrap(), Fps { num: 30, den: 1 });
        assert!(Fps::parse("0/1").is_err() && Fps::parse("abc").is_err());
    }

    #[test]
    fn seconds_are_exact_decimals() {
        let f30 = Fps::new(30, 1).unwrap();
        assert_eq!(seconds_to_frame("3.0", f30).unwrap(), 90);
        assert_eq!(seconds_to_frame("1.5", f30).unwrap(), 45);
        assert_eq!(seconds_to_frame("0.1", f30).unwrap(), 3);
        assert_eq!(seconds_to_frame("120", Fps::new(30000, 1001).unwrap()).unwrap(), 3596);
        assert_eq!(seconds_to_frame(&f64_decimal(0.1), f30).unwrap(), 3);
        assert_eq!(seconds_to_frame("1e1", f30).unwrap(), 300);
    }
}
