use std::cell::Cell;
use std::fmt;
use std::time::{Duration, Instant};

use crate::mem;

thread_local!(static TIME_DEPTH: Cell<usize> = Cell::new(0));

pub struct HumanDuration(Duration);
impl HumanDuration {
    pub fn since(i: Instant) -> Self {
        Self(Instant::now().duration_since(i))
    }
}
impl From<Duration> for HumanDuration {
    fn from(d: Duration) -> Self {
        Self(d)
    }
}
impl fmt::Display for HumanDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.0.as_secs();
        let alt = f.alternate();
        macro_rules! try_unit {
            ($secs:expr, $sg:expr, $pl:expr, $s:expr) => {
                let cnt = t / $secs;
                if cnt == 1 {
                    if alt {
                        return write!(f, "{}{}", cnt, $s);
                    } else {
                        return write!(f, "{} {}", cnt, $sg);
                    }
                } else if cnt > 1 {
                    if alt {
                        return write!(f, "{}{}", cnt, $s);
                    } else {
                        return write!(f, "{} {}", cnt, $pl);
                    }
                }
            };
        }

        if t > 0 {
            try_unit!(365 * 24 * 60 * 60, "year", "years", "y");
            try_unit!(7 * 24 * 60 * 60, "week", "weeks", "w");
            try_unit!(24 * 60 * 60, "day", "days", "d");
            try_unit!(60 * 60, "hour", "hours", "h");
            try_unit!(60, "minute", "minutes", "m");
            try_unit!(1, "second", "seconds", "s");
        } else {
            // Time was too precise for the standard path, use millis
            let t = self.0.as_millis();
            if t > 0 {
                return write!(f, "{}{}", t, if alt { "ms" } else { " milliseconds" });
            }
        }
        write!(f, "0{}", if alt { "s" } else { " seconds" })
    }
}

pub fn time<T, F>(timing_enabled: bool, what: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    if !timing_enabled {
        return f();
    }

    let old = TIME_DEPTH.with(|slot| {
        let r = slot.get();
        slot.set(r + 1);
        r
    });

    let start = Instant::now();
    let rv = f();
    let dur = start.elapsed();

    print_time_passes_entry(true, what, dur);

    TIME_DEPTH.with(|slot| slot.set(old));

    rv
}

pub fn print_time_passes_entry(timing_enabled: bool, what: &str, dur: Duration) {
    if !timing_enabled {
        return;
    }

    let indentation = TIME_DEPTH.with(|slot| slot.get());

    let mem_string = match mem::get_resident_size() {
        Some(n) => {
            let mb = n as f64 / 1_000_000.0;
            format!("; rss: {}MB", mb.round() as usize)
        }
        None => String::new(),
    };
    println!(
        "{}time: {}{}\t{}",
        "  ".repeat(indentation),
        duration_to_secs_str(dur),
        mem_string,
        what
    );
}

// Hack up our own formatting for the duration to make it easier for scripts
// to parse (always use the same number of decimal places and the same unit).
pub fn duration_to_secs_str(dur: std::time::Duration) -> String {
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;
    let secs = dur.as_secs() as f64 + dur.subsec_nanos() as f64 / NANOS_PER_SEC;

    format!("{:.3}", secs)
}
