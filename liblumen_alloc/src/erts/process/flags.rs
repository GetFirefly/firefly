use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};
use core::sync::atomic::{AtomicU32, Ordering};

/// This type provides an enum-like type with bitflag semantics,
/// i.e. you can use the `!`, `|`, `&`, `|=`, and `&=` operators
/// to combine multiple flags in one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ProcessFlags(u32);
impl ProcessFlags {
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
    /// This flag indicates the processes linked to this process should send exit messages instead
    /// of causing this process to exit when they exit
    pub const TrapExit: Self = Self(1 << 6);

    pub fn are_set(&self, flags: ProcessFlags) -> bool {
        (*self & flags) == flags
    }
}
impl Into<u32> for ProcessFlags {
    #[inline]
    fn into(self) -> u32 {
        self.0
    }
}
impl From<u32> for ProcessFlags {
    #[inline]
    fn from(n: u32) -> Self {
        Self(n)
    }
}
impl Not for ProcessFlags {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl BitOr for ProcessFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for ProcessFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0
    }
}
impl BitAnd for ProcessFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl BitAndAssign for ProcessFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0
    }
}

/// This type is a wrapper around `AtomicU32` and provides atomic
/// semantics for the `ProcessFlags` enum type
#[repr(transparent)]
pub struct AtomicProcessFlags(AtomicU32);
impl AtomicProcessFlags {
    /// Create a new `AtomicProcessFlags`
    #[inline]
    pub fn new(flag: ProcessFlags) -> Self {
        Self(AtomicU32::new(flag.into()))
    }

    /// Fetch the current value, with the given `Ordering`
    #[inline]
    pub fn load(&self, ordering: Ordering) -> ProcessFlags {
        self.0.load(ordering).into()
    }

    /// Fetch the current value, using `Relaxed` ordering
    #[inline]
    pub fn get(&self) -> ProcessFlags {
        self.0.load(Ordering::Relaxed).into()
    }

    /// Check if the current value contains the given flags, uses `Relaxed` ordering
    #[inline]
    pub fn are_set(&self, flags: ProcessFlags) -> bool {
        self.get() & flags == flags
    }

    /// Set the given flags, uses `AcqRel` ordering
    #[inline]
    pub fn set(&self, flags: ProcessFlags) -> ProcessFlags {
        self.0.fetch_or(flags.into(), Ordering::AcqRel).into()
    }

    /// Clear the given flags, uses `AcqRel` ordering
    #[inline]
    pub fn clear(&self, flags: ProcessFlags) -> ProcessFlags {
        let cleared = !flags;
        self.0.fetch_and(cleared.into(), Ordering::AcqRel).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_flag_test() {
        let mut flags = ProcessFlags::Default;
        // Set fullsweep
        flags |= ProcessFlags::NeedFullSweep;
        assert!(flags & ProcessFlags::NeedFullSweep == ProcessFlags::NeedFullSweep);
        // Set force_gc
        flags |= ProcessFlags::ForceGC;
        assert!(flags & ProcessFlags::NeedFullSweep == ProcessFlags::NeedFullSweep);
        assert!(flags & ProcessFlags::ForceGC == ProcessFlags::ForceGC);
        // Ensure we can check multiple flags at once
        let checking = ProcessFlags::ForceGC | ProcessFlags::NeedFullSweep;
        assert!(flags & checking == checking);
        // Clear force_gc
        flags &= !ProcessFlags::ForceGC;
        assert!(flags & ProcessFlags::ForceGC != ProcessFlags::ForceGC);
        assert!(flags & ProcessFlags::NeedFullSweep == ProcessFlags::NeedFullSweep);
    }

    #[test]
    fn atomic_process_flag_test() {
        let flags = AtomicProcessFlags::new(ProcessFlags::Default);
        flags.set(ProcessFlags::NeedFullSweep);
        assert!(flags.are_set(ProcessFlags::NeedFullSweep));
        flags.set(ProcessFlags::ForceGC);
        assert!(flags.are_set(ProcessFlags::NeedFullSweep));
        assert!(flags.are_set(ProcessFlags::ForceGC));
        assert!(flags.are_set(ProcessFlags::ForceGC | ProcessFlags::NeedFullSweep));
        flags.clear(ProcessFlags::ForceGC);
        assert!(flags.are_set(ProcessFlags::NeedFullSweep));
        assert!(!flags.are_set(ProcessFlags::ForceGC));
    }
}
