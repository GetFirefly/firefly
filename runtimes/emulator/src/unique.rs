use std::mem::{self, ManuallyDrop};
use std::sync::atomic::{AtomicI64, Ordering};

use firefly_number::Int;

static mut UNIQUE_DATA: UniqueData = UniqueData::default();
static UNIQUE_MONOTONIC_DATA: AtomicI64 = AtomicI64::new(-1);

const UNIQUE_MONOTONIC_OFFSET: i64 = Int::MIN_SMALL;

/// Initialize the global unique integer state
pub fn init(
    num_schedulers: usize,
    num_dirty_cpu_schedulers: usize,
    num_dirty_io_schedulers: usize,
) {
    let mut val0_max = num_schedulers as u64;
    val0_max += num_dirty_cpu_schedulers as u64;
    val0_max += num_dirty_io_schedulers as u64;
    let bits = fit_in_bits_i64(val0_max as i64);
    unsafe {
        UNIQUE_DATA.r.val0_max = val0_max;
        UNIQUE_DATA.r.left_shift = bits;
        UNIQUE_DATA.r.right_shift = 64 - bits;
        UNIQUE_DATA.r.mask = (1 << bits as u64) - 1;
        UNIQUE_DATA.w.store(-1, Ordering::Relaxed);
    }
}

/// Get the next unique integer value from the global state
///
/// If `positive` is true, the integer will be a positive value
#[allow(unused)]
pub fn get_unique_integer(positive: bool) -> Int {
    let value1 = unsafe { UNIQUE_DATA.w.fetch_add(1, Ordering::Acquire) };
    make_unique_integer(0, value1 as u64, positive)
}

/// Get the next raw unique monotonic integer from the global state
pub fn get_raw_unique_monotonic_integer() -> u64 {
    UNIQUE_MONOTONIC_DATA.fetch_add(1, Ordering::Acquire) as u64
}

/// Compute the next unique monotonic integer value from the given value
///
/// If `positive` is true, the integer will be a positive value
pub fn get_unique_monotonic_integer(positive: bool) -> Int {
    let raw = get_raw_unique_monotonic_integer();
    if positive {
        (raw + 1).into()
    } else {
        let value = (raw as i64) + UNIQUE_MONOTONIC_OFFSET;
        value.into()
    }
}

/// Make a unique integer from raw parts
pub fn make_unique_integer(value0: u64, value1: u64, positive: bool) -> Int {
    let mut unique = [0u64; 2];
    unique[0] = value0;
    unique[0] |= value1 << unsafe { UNIQUE_DATA.r.left_shift as u64 };
    unique[1] = value1 >> unsafe { UNIQUE_DATA.r.right_shift as u64 };
    unique[1] &= unsafe { UNIQUE_DATA.r.mask };

    if positive {
        unique[0] += 1;
        if unique[0] == 0 {
            unique[1] += 1;
        }
    } else {
        if unique[1] == 0 && unique[0] < (-1 * (Int::MIN_SMALL as i64)) as u64 {
            let mut s_unique = unique[0] as i64;
            s_unique += Int::MIN_SMALL;
            assert!(Int::MIN_SMALL <= s_unique && s_unique < 0);
            return s_unique.into();
        }
        if unique[0] < (-1 * Int::MIN_SMALL) as u64 {
            assert!(unique[1] != 0);
            unique[1] -= 1;
        }
        unique[0] += Int::MIN_SMALL as u64;
    }

    if unique[1] == 0 {
        unique[0].into()
    } else {
        i128::from_be_bytes(unsafe { mem::transmute::<_, [u8; 16]>(unique) }).into()
    }
}

union UniqueData {
    r: UniqueDataParts,
    w: ManuallyDrop<AtomicI64>,
}
impl const Default for UniqueData {
    fn default() -> Self {
        let mut data = Self {
            r: UniqueDataParts::default(),
        };
        data.w = ManuallyDrop::new(AtomicI64::new(0));
        data
    }
}

#[derive(Copy, Clone)]
struct UniqueDataParts {
    left_shift: i32,
    right_shift: i32,
    mask: u64,
    val0_max: u64,
}
impl const Default for UniqueDataParts {
    fn default() -> Self {
        Self {
            left_shift: 0,
            right_shift: 0,
            mask: 0,
            val0_max: 0,
        }
    }
}

#[inline]
fn fit_in_bits_i64(value: i64) -> i32 {
    fit_in_bits(value, 5)
}

fn fit_in_bits(mut value: i64, start: u32) -> i32 {
    // (mask, bits)
    const FIB_DATA: [(i64, i32); 6] = [
        (0x2u64 as i64, 1),
        (0xcu64 as i64, 2),
        (0xf0u64 as i64, 4),
        (0xff00u64 as i64, 8),
        (0xffff0000u64 as i64, 16),
        (0xffffffff00000000u64 as i64, 32),
    ];

    let mut bits = 0;

    for i in (0..=(start as usize)).rev() {
        if value & FIB_DATA[i].0 > 0 {
            value >>= FIB_DATA[i].1;
            bits |= FIB_DATA[i].1;
        }
    }

    bits + 1
}
