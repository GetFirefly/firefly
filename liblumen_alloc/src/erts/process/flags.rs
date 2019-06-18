use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};
use core::sync::atomic::{AtomicU32, Ordering};

/// This type provides an enum-like type with bitflag semantics,
/// i.e. you can use the `!`, `|`, `&`, `|=`, and `&=` operators
/// to combine multiple flags in one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ProcessFlag(u32);
impl ProcessFlag {
    #![allow(non_upper_case_globals)]

    /// The default value where no flags are set
    pub const Default: Self = Self(1 << 0);
    /// This flag indicates that the next GC should grow the heap unconditionally
    pub const GrowHeap: Self = Self(1 << 1);
    /// This flag indicates that the next GC should be a full sweep unconditionally
    pub const NeedFullSweep: Self = Self(1 << 2);
    /// This flag indicates that the next GC check will always return `true`,
    /// i.e. a collection will be forced
    pub const ForceGC: Self = Self(1 << 3);
    /// This flag indicates that GC should be disabled temporarily
    pub const DisableGC: Self = Self(1 << 4);
    /// This flag indicates that GC should be delayed temporarily
    pub const DelayGC: Self = Self(1 << 5);

    // Internal value used to validate conversions from raw u32 values
    const MAX_VALUE: u32 = 1 << 5;
}
impl Into<u32> for ProcessFlag {
    #[inline]
    fn into(self) -> u32 {
        self.0
    }
}
impl From<u32> for ProcessFlag {
    #[inline]
    fn from(n: u32) -> Self {
        assert!(n <= Self::MAX_VALUE);
        Self(n)
    }
}
impl Not for ProcessFlag {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl BitOr for ProcessFlag {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for ProcessFlag {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0
    }
}
impl BitAnd for ProcessFlag {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl BitAndAssign for ProcessFlag {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0
    }
}

/// This type is a wrapper around `AtomicU32` and provides atomic
/// semantics for the `ProcessFlag` enum type
#[repr(transparent)]
pub struct AtomicProcessFlag(AtomicU32);
impl AtomicProcessFlag {
    /// Create a new `AtomicProcessFlag`
    #[inline]
    pub fn new(flag: ProcessFlag) -> Self {
        Self(AtomicU32::new(flag.into()))
    }

    /// Fetch the current value, with the given `Ordering`
    #[inline]
    pub fn load(&self, ordering: Ordering) -> ProcessFlag {
        self.0.load(ordering).into()
    }

    /// Fetch the current value, using `Relaxed` ordering
    #[inline]
    pub fn get(&self) -> ProcessFlag {
        self.0.load(Ordering::Relaxed).into()
    }

    /// Check if the current value contains the given flags, uses `Relaxed` ordering
    #[inline]
    pub fn is_set(&self, flags: ProcessFlag) -> bool {
        self.get() & flags == flags
    }

    /// Set the given flags, uses `AcqRel` ordering
    #[inline]
    pub fn set(&self, flags: ProcessFlag) -> ProcessFlag {
        self.0.fetch_or(flags.into(), Ordering::AcqRel).into()
    }

    /// Clear the given flags, uses `AcqRel` ordering
    #[inline]
    pub fn clear(&self, flags: ProcessFlag) -> ProcessFlag {
        let cleared = !flags;
        self.0.fetch_and(cleared.into(), Ordering::AcqRel).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_flag_test() {
        let mut flags = ProcessFlag::Default;
        // Set fullsweep
        flags |= ProcessFlag::NeedFullSweep;
        assert!(flags & ProcessFlag::NeedFullSweep == ProcessFlag::NeedFullSweep);
        // Set force_gc
        flags |= ProcessFlag::ForceGC;
        assert!(flags & ProcessFlag::NeedFullSweep == ProcessFlag::NeedFullSweep);
        assert!(flags & ProcessFlag::ForceGC == ProcessFlag::ForceGC);
        // Ensure we can check multiple flags at once
        let checking = ProcessFlag::ForceGC | ProcessFlag::NeedFullSweep;
        assert!(flags & checking == checking);
        // Clear force_gc
        flags &= !ProcessFlag::ForceGC;
        assert!(flags & ProcessFlag::ForceGC != ProcessFlag::ForceGC);
        assert!(flags & ProcessFlag::NeedFullSweep == ProcessFlag::NeedFullSweep);
    }

    #[test]
    fn atomic_process_flag_test() {
        let flags = AtomicProcessFlag::new(ProcessFlag::Default);
        flags.set(ProcessFlag::NeedFullSweep);
        assert!(flags.is_set(ProcessFlag::NeedFullSweep));
        flags.set(ProcessFlag::ForceGC);
        assert!(flags.is_set(ProcessFlag::NeedFullSweep));
        assert!(flags.is_set(ProcessFlag::ForceGC));
        assert!(flags.is_set(ProcessFlag::ForceGC | ProcessFlag::NeedFullSweep));
        flags.clear(ProcessFlag::ForceGC);
        assert!(flags.is_set(ProcessFlag::NeedFullSweep));
        assert!(!flags.is_set(ProcessFlag::ForceGC));
    }
}
