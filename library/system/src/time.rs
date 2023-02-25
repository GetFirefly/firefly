pub use core::time::Duration;

pub use crate::arch::time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH};

use core::fmt;
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Add, AddAssign, Sub, SubAssign};

/// This type represents a timestamp derived from the system monotonic clock.
///
/// The actual value of the timestamp is relative to the first time the monotonic clock was read when
/// the system started, and is always monotonically increasing.
///
/// A monotonic time is always tracked at nanosecond precision, but it is not guaranteed that
/// the monotonic clock actually is that precise.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotonicTime(Instant);
impl MonotonicTime {
    /// Get the current monotonic time
    pub fn now() -> Self {
        Self(Instant::now())
    }

    /// Get the monotonic time at which `timeout` will expire
    ///
    /// If this returns `None`, the timeout will never occur.
    pub fn after(timeout: Timeout) -> Option<Self> {
        if timeout.is_immediate() {
            None
        } else {
            Some(Self::now() + timeout)
        }
    }

    /// Returns the time that has elapsed between this timestamp and the start of the monotonic clock
    pub fn elapsed(&self) -> Duration {
        instant_to_duration(self.0)
    }

    /// Returns the time that has elapsed between this timestamp and `earlier`
    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.0.duration_since(earlier.0)
    }

    /// Returns a u64 representing this monotonic time
    ///
    /// The actual meaning of the bits may differ between platforms, but
    /// conceptually we convert the underlying `Instant` to a `Duration`, and
    /// then get the number of seconds represented by the `Duration`.
    pub fn as_u64(&self) -> u64 {
        self.elapsed().as_secs()
    }

    /// Returns a u64 representing this monotonic time as microseconds
    ///
    /// If the duration between `now` and the start of the monotonic clock is too large
    /// to fit in a u64, then the value will be truncated, but that is never expected to
    /// happen, as the amount of time that would be represented by such a value is absurd,
    /// on the order of millions of years.
    pub fn as_usecs(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }
}
impl fmt::Display for MonotonicTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.as_u64())
    }
}
impl From<Instant> for MonotonicTime {
    fn from(instant: Instant) -> Self {
        Self(instant)
    }
}
impl Add<Duration> for MonotonicTime {
    type Output = MonotonicTime;

    fn add(self, other: Duration) -> Self::Output {
        Self(self.0 + other)
    }
}
impl Add<Timeout> for MonotonicTime {
    type Output = MonotonicTime;

    fn add(self, other: Timeout) -> Self::Output {
        Self(self.0 + other.as_duration())
    }
}
impl AddAssign<Duration> for MonotonicTime {
    fn add_assign(&mut self, other: Duration) {
        self.0 += other
    }
}
impl AddAssign<Timeout> for MonotonicTime {
    fn add_assign(&mut self, other: Timeout) {
        self.0 += other.as_duration()
    }
}
impl Sub<MonotonicTime> for MonotonicTime {
    type Output = Duration;

    fn sub(self, other: Self) -> Self::Output {
        self.duration_since(other)
    }
}
impl SubAssign<Duration> for MonotonicTime {
    fn sub_assign(&mut self, other: Duration) {
        self.0 -= other;
    }
}

/// Occurs when attempting to convert a value to a [`Timeout`]
#[derive(Debug, Copy, Clone)]
pub struct InvalidTimeoutError;
impl fmt::Display for InvalidTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "invalid timeout value, must be the atom 'infinity', or a non-negative integer value"
        )
    }
}

#[cfg(any(unix, windows, target_os = "wasi", target_family = "wasm"))]
impl std::error::Error for InvalidTimeoutError {}

/// Represents the duration until an operation should time out
///
/// Internally this is represented by a [`Duration`], with boundary values of
/// the duration representing immediate and infinite values respectively.
///
/// We choose to wrap [`Duration`] like this because the semantics around timeouts
/// are a bit different, and we can provide a few niceties that you don't get out of
/// the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeout(Duration);
impl Timeout {
    pub const IMMEDIATE: Self = Self(Duration::ZERO);
    pub const INFINITY: Self = Self(Duration::MAX);

    /// Returns true if this timeout occurs immediately
    #[inline]
    pub fn is_immediate(&self) -> bool {
        self.0.is_zero()
    }

    /// Returns true if this timeout will never be reached
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.0 == Self::INFINITY
    }

    /// Constructs a new timeout from the given [`Duration`] value.
    #[inline]
    pub const fn new(duration: Duration) -> Self {
        Self(duration)
    }

    /// Construct a new timeout from a duration in milliseconds
    ///
    /// See also the `From` and `TryFrom` implementations for `u64` and `i64` respectively.
    #[inline]
    pub fn from_millis(millis: u64) -> Self {
        millis.into()
    }

    /// Convert this timeout to an equivalent [`Duration`] value
    pub fn as_duration(&self) -> Duration {
        self.0
    }
}
impl Add<Duration> for Timeout {
    type Output = Timeout;

    fn add(self, other: Duration) -> Self::Output {
        Self(self.0.saturating_add(other))
    }
}
impl AddAssign<Duration> for Timeout {
    fn add_assign(&mut self, other: Duration) {
        self.0 = self.0.saturating_add(other);
    }
}
impl Sub<Duration> for Timeout {
    type Output = Timeout;

    fn sub(self, other: Duration) -> Self::Output {
        Self(self.0.saturating_sub(other))
    }
}
impl SubAssign<Duration> for Timeout {
    fn sub_assign(&mut self, other: Duration) {
        self.0 = self.0.saturating_sub(other);
    }
}
impl From<Duration> for Timeout {
    fn from(duration: Duration) -> Self {
        Self(duration)
    }
}
impl Into<Duration> for Timeout {
    fn into(self) -> Duration {
        self.0
    }
}
impl PartialEq<Duration> for Timeout {
    fn eq(&self, other: &Duration) -> bool {
        self.0.eq(other)
    }
}
impl PartialEq<Timeout> for Duration {
    fn eq(&self, other: &Timeout) -> bool {
        self.eq(&other.0)
    }
}
impl PartialOrd<Duration> for Timeout {
    fn partial_cmp(&self, other: &Duration) -> Option<core::cmp::Ordering> {
        Some(self.0.cmp(other))
    }
}
impl PartialOrd<Timeout> for Duration {
    fn partial_cmp(&self, other: &Timeout) -> Option<core::cmp::Ordering> {
        Some(self.cmp(&other.0))
    }
}
impl From<u64> for Timeout {
    fn from(millis: u64) -> Self {
        Self(Duration::from_millis(millis))
    }
}
impl TryFrom<i64> for Timeout {
    type Error = InvalidTimeoutError;

    fn try_from(millis: i64) -> Result<Self, Self::Error> {
        Ok(Self(Duration::from_millis(
            millis.try_into().map_err(|_| InvalidTimeoutError)?,
        )))
    }
}
impl Default for Timeout {
    #[inline]
    fn default() -> Self {
        Self::INFINITY
    }
}

/// Represents units in which time values can be represented
///
/// All of these units are well-defined relative to a second of real time.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TimeUnit {
    /// Represents a precise frequency, where the given number of units occurs every second of real time
    ///
    /// For example, `TimeUnit::Hertz(1_000)` is equivalent to `TimeUnit::Millisecond`.
    Hertz(NonZeroUsize),
    /// Equivalent to `TimeUnit::Hertz(1)`
    Second,
    /// A thousandth of a second. Equivalent to `TimeUnit::Hertz(1_000)`
    Millisecond,
    /// A millionth of a second. Equivalent to `TimeUnit::Hertz(1_000_000)`
    Microsecond,
    /// A billionth of a second. Equivalent to `TimeUnit::Hertz(1_000_000_000)`
    Nanosecond,
    /// A unit of time that is platform-specific.
    ///
    /// While this is equivalent to nanoseconds on most platforms, some platforms
    /// do not provide for such high resolution clocks, such as WebAssembly in a browser
    /// environment. In those cases, the unit might be much less precise.
    Native,
    /// A unit corresponding that used by the performance counters on the current system.
    ///
    /// Like `Native`, this is generally nanoseconds, but may be much less precise on some systems.
    PerformanceCounter,
}
impl TimeUnit {
    const SECOND: usize = 1;
    const MILLISECOND: usize = Self::SECOND * 1_000;
    const MICROSECOND: usize = Self::MILLISECOND * 1_000;
    const NANOSECOND: usize = Self::MICROSECOND * 1_000;

    /// Create a new unit expressed in a given frequency, or units per second of real time
    ///
    /// The given value will be normalized, so if it corresponds to a well-known unit, the result will be that unit
    pub fn new(hertz: NonZeroUsize) -> Self {
        match hertz.get() {
            Self::SECOND => Self::Second,
            Self::MILLISECOND => Self::Millisecond,
            Self::MICROSECOND => Self::Microsecond,
            Self::NANOSECOND => Self::Nanosecond,
            _ => Self::Hertz(hertz),
        }
    }

    /// Return the effective frequency, or units per second of real time, of this unit.
    pub fn hertz(&self) -> usize {
        match self {
            Self::Hertz(hertz) => hertz.get(),
            Self::Second => Self::SECOND,
            Self::Millisecond => Self::MILLISECOND,
            Self::Microsecond => Self::MICROSECOND,
            Self::Nanosecond => Self::NANOSECOND,
            // As a side-channel protection browsers limit most counters to 1 millisecond resolution
            #[cfg(target_family = "wasm")]
            Self::Native | Self::PerformanceCounter => Self::MILLISECOND,
            #[cfg(not(target_family = "wasm"))]
            Self::Native | Self::PerformanceCounter => Self::NANOSECOND,
        }
    }
}
impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Hertz(hertz) => write!(f, "@{}Hz", hertz.get()),
            Self::Second => write!(f, "s"),
            Self::Millisecond => write!(f, "ms"),
            Self::Microsecond => write!(f, "us"),
            Self::Nanosecond => write!(f, "ns"),
            #[cfg(target_arch = "wasm32")]
            Self::Native | Self::PerformanceCounter => write!(f, "ms"),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Native | Self::PerformanceCounter => write!(f, "ns"),
        }
    }
}
impl core::str::FromStr for TimeUnit {
    type Err = TimeUnitConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "second" | "seconds" | "s" => Ok(Self::Second),
            "millisecond" | "milliseconds" | "ms" => Ok(Self::Millisecond),
            "microsecond" | "microseconds" | "us" => Ok(Self::Microsecond),
            "nanosecond" | "nanoseconds" | "ns" => Ok(Self::Nanosecond),
            "native" => Ok(Self::Native),
            "perf_counter" => Ok(Self::PerformanceCounter),
            _ => Err(TimeUnitConversionError::InvalidUnitName),
        }
    }
}
impl TryFrom<usize> for TimeUnit {
    type Error = TimeUnitConversionError;

    fn try_from(hertz: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(hertz)
            .map(Self::new)
            .ok_or(TimeUnitConversionError::InvalidHertzValue)
    }
}
impl TryFrom<i64> for TimeUnit {
    type Error = TimeUnitConversionError;

    fn try_from(hertz: i64) -> Result<Self, Self::Error> {
        use core::num::NonZeroI64;

        let hertz = NonZeroI64::new(hertz).ok_or(TimeUnitConversionError::InvalidHertzValue)?;
        hertz
            .try_into()
            .map(Self::new)
            .map_err(|_| TimeUnitConversionError::InvalidHertzValue)
    }
}

/// Represents an invalid conversion from some type to [`TimeUnit`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeUnitConversionError {
    InvalidHertzValue,
    InvalidUnitName,
    InvalidConversion,
}
impl fmt::Display for TimeUnitConversionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidHertzValue => write!(f, "invalid conversion to hertz: expected value > 0"),
            Self::InvalidUnitName => write!(f, "invalid unit name, expected one of: second(s), millisecond(s), microsecond(s), nanosecond(s), native, or perf_counter"),
            Self::InvalidConversion => write!(f, "invalid unit conversion: unsupported source type"),
        }
    }
}

#[cfg(windows)]
pub fn instant_to_duration(instant: Instant) -> Duration {
    // On windows, `Instant` is a wrapper around `Duration`
    //
    // That duration value is relative to an arbitrary microsecond epoch from
    // the winapi QueryPerformanceCounter function.
    struct Repr {
        t: Duration,
    }

    mem::transmute::<Instant, Repr>(instant).t
}

#[cfg(any(
    all(target_os = "macos", any(not(target_arch = "aarch64"))),
    target_os = "ios",
    target_os = "watchos",
))]
pub fn instant_to_duration(instant: Instant) -> Duration {
    use core::sync::atomic::{AtomicU64, Ordering};

    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    struct mach_timebase_info {
        numer: u32,
        denom: u32,
    }

    struct Repr {
        t: u64,
    }

    extern "C" {
        fn mach_timebase_info(info: *mut mach_timebase_info) -> libc::c_int;
    }

    // Used to cache the conversion info for mach ticks
    static INFO_BITS: AtomicU64 = AtomicU64::new(0);

    // Load the conversion info
    let mut info = mach_timebase_info::default();
    let info_bits = INFO_BITS.load(Ordering::Relaxed);
    if info_bits != 0 {
        info.numer = info_bits as u32;
        info.denom = (info_bits >> 32) as u32;
    } else {
        unsafe {
            mach_timebase_info(&mut info);
        }
        INFO_BITS.store(
            ((info.denom as u64) << 32) | (info.numer as u64),
            Ordering::Relaxed,
        );
    }

    // Get the raw instant value
    let value = unsafe { mem::transmute::<Instant, Repr>(instant).t };
    // Convert it to nanoseconds and return as a Duration
    let nanos = {
        let numer = info.numer as u64;
        let denom = info.denom as u64;
        let q = value / denom;
        let r = value % denom;
        q * numer + r * numer / denom
    };

    Duration::new(nanos / (1_000_000_000), (nanos % 1_000_000_000) as u32)
}

#[cfg(all(
    unix,
    not(any(
        all(target_os = "macos", any(not(target_arch = "aarch64"))),
        target_os = "ios",
        target_os = "watchos"
    ))
))]
pub fn instant_to_duration(instant: Instant) -> Duration {
    struct Timespec {
        tv_sec: u64,
        tv_nsec: u32,
    }

    struct Repr {
        t: Timespec,
    }

    let value = unsafe { mem::transmute::<Instant, Repr>(instant).t };
    Duration::new(value.tv_sec, value.tv_nsec)
}

#[cfg(any(
    target_family = "wasm",
    target_family = "wasi",
    not(any(unix, windows))
))]
pub fn instant_to_duration(instant: Instant) -> Duration {
    // On wasm/wasi targets, and unsupported platforms, Instant is a transparent wrapper around Duration
    mem::transmute::<Instant, Duration>(instant)
}
