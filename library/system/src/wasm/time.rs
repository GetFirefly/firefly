pub use core::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(transparent)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(transparent)]
pub struct SystemTime(Duration);

/// A `no_std`-compatible `std::time::SystemTimeError`
#[derive(Clone, Debug)]
pub struct SystemTimeError(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Self {
        let window = web_sys::window().unwrap();
        let ms = window.performance().unwrap().now();
        let secs = (ms as u64) / 1_000;
        let nanos = (((ms as u64) % 1_000) as u32) * 1_000_000;
        Self(Duration::new(secs, nanos))
    }

    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    pub fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0)
    }

    pub fn saturating_duration_since(&self, earlier: Self) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    pub fn elapsed(&self) -> Duration {
        Self::now() - *self
    }

    pub fn checked_add(&self, duration: Duration) -> Option<Self> {
        Some(Self(self.0.checked_add(duration)?))
    }

    pub fn checked_sub(&self, duration: Duration) -> Option<Self> {
        Some(Self(self.0.checked_sub(duration)?))
    }
}
impl fmt::Debug for Instant {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}
impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, other: Duration) -> Instant {
        self.checked_add(other)
            .expect("overflow when adding duration to instant")
    }
}
impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}
impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        self.checked_sub(other)
            .expect("underflow when subtracting duration from instant")
    }
}
impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}
impl Sub<Instant> for Instant {
    type Output = Duration;

    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}

impl SystemTime {
    pub fn now() -> Self {
        UNIX_EPOCH + Duration::from_millis(js_sys::Date::now() as u64)
    }

    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0)
    }

    pub fn elapsed(&self) -> Duration {
        Self::now() - *self
    }

    pub fn checked_add(&self, duration: Duration) -> Option<Self> {
        Some(Self(self.0.checked_add(duration)?))
    }

    pub fn checked_sub(&self, duration: Duration) -> Option<Self> {
        Some(Self(self.0.checked_sub(duration)?))
    }
}

impl fmt::Debug for SystemTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}
impl Add<Duration> for SystemTime {
    type Output = SystemTime;

    fn add(self, other: Duration) -> Self {
        self.checked_add(other)
            .expect("overflow when adding duration to instant")
    }
}
impl AddAssign<Duration> for SystemTime {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}
impl Sub<Duration> for SystemTime {
    type Output = SystemTime;

    fn sub(self, other: Duration) -> Self {
        self.checked_sub(other)
            .expect("underflow when subtracting duration from instant")
    }
}
impl SubAssign<Duration> for SystemTime {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

impl SystemTimeError {
    pub fn duration(&self) -> Duration {
        self.0
    }
}
impl fmt::Display for SystemTimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "second time provided was later than self")
    }
}

fn now() -> Duration {}
