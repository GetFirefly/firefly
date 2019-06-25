//! This module provides a histogram view over a stream of samples.
//! 
//! The histogram itself is defined via the `define_histogram!` macro,
//! which takes the number of buckets that the histogram should use, and
//! then generates a module which will use a constant amount of space for
//! the histogram, and can be constructed at compile-time.
//!
//! # Examples
//! 
//! ```
//! use liblumen_alloc::stats::Histogram;
//! 
//! mod stats {
//!     use liblumen_alloc::define_histogram;
//!     define_histogram!(10);
//! }
//! # fn main() {
//! // Create a histogram that will spread the given range over 
//! // the 10 buckets the `Histogram` type was defined with.
//! let mut histogram = stats::Histogram::with_const_width(0, 10_000);
//!
//! // Adds some samples to the histogram.
//! for sample in 0..100 {
//!     histogram.add(sample);
//!     histogram.add(sample * sample);
//! }
//!
//! // Iterate over buckets and do stuff with their range and count.
//! for ((start, end), count) in histogram.into_iter() {
//!     println!("{}..{} has {} samples", start, end, count);
//! }
//!
//! // And you can also `Display` a histogram!
//! println!("{}", histogram);
//!
//! // Prints:
//! //
//! // ```
//! // # Number of samples = 200
//! // # Min = 0
//! // # Max = 9801
//! // #
//! // # Mean = 1666.5000000000005
//! // # Standard deviation = 2641.2281518263426
//! // # Variance = 6976086.1499999985
//! // #
//! // # Each ∎ is a count of 2
//! // #
//! //    0 ..  980 [ 132 ]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
//! //  980 .. 1960 [  13 ]: ∎∎∎∎∎∎
//! // 1960 .. 2940 [  10 ]: ∎∎∎∎∎
//! // 2940 .. 3920 [   8 ]: ∎∎∎∎
//! // 3920 .. 4900 [   7 ]: ∎∎∎
//! // 4900 .. 5880 [   7 ]: ∎∎∎
//! // 5880 .. 6860 [   6 ]: ∎∎∎
//! // 6860 .. 7840 [   6 ]: ∎∎∎
//! // 7840 .. 8820 [   5 ]: ∎∎
//! // 8820 .. 9800 [   6 ]: ∎∎∎
//! // ```
//! # }
//! ```
#![deny(unsafe_code)]

use core::fmt;

pub use defaults::Histogram as DefaultHistogram;

/// This trait represents the bare minimum functionality needed
/// to interact with an implementation by a mutator
pub trait Histogram: fmt::Display {
    /// Add a sample to the histogram.
    ///
    /// Fails if the sample is out of range of the histogram.
    fn add(&mut self, x: u64) -> Result<(), ()>;
}

/// This macro creates a `Histogram` type in the same module which
/// will use `$LEN` buckets internally for sample data
#[macro_export]
macro_rules! define_histogram {
    ($LEN:expr) => {
        /// A histogram with a number of bins known at compile time.
        #[derive(Clone)]
        pub struct Histogram {
            /// The ranges defining the bins of the histogram.
            range: [u64; $LEN + 1],
            /// The bins of the histogram.
            bin: [u64; $LEN],
            /// Online statistics like mean, variance, etc.
            stats: $crate::stats::OnlineStats,
            /// Min/max values
            minmax: $crate::stats::MinMax<u64>,
        }


        impl core::fmt::Debug for Histogram {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                f.write_str("Histogram {{ range: ")?;
                self.range[..].fmt(f)?;
                f.write_str(", bin: ")?;
                self.bin[..].fmt(f)?;
                f.write_fmt(format_args!(", stats: {:?}", &self.stats))?;
                f.write_fmt(format_args!(", minmax: {:?}", &self.minmax))?;
                f.write_str(" }}")
            }
        }

        impl Histogram {
            /// The number of bins of the histogram.
            const LEN: usize = $LEN;

            /// Construct a histogram with constant bin width.
            #[inline]
            pub fn with_const_width(start: u64, end: u64) -> Self {
                let step = (end - start) / (Self::LEN as u64);
                let mut range = [0; Self::LEN + 1];
                for (i, r) in range.iter_mut().enumerate() {
                    *r = start + step * (i as u64);
                }

                Self {
                    range,
                    bin: [0; Self::LEN],
                    stats: $crate::stats::OnlineStats::default(),
                    minmax: $crate::stats::MinMax::default(),
                }
            }

            /// Construct a histogram from given ranges.
            ///
            /// The ranges are given by an iterator of floats where neighboring
            /// pairs `(a, b)` define a bin for all `x` where `a <= x < b`.
            ///
            /// Fails if the iterator is too short (less than `n + 1` where `n`
            /// is the number of bins), is not sorted or contains `nan`. `inf`
            /// and empty ranges are allowed.
            #[inline]
            pub fn from_ranges<T>(ranges: T) -> Result<Self, ()>
                where T: IntoIterator<Item = u64>
            {
                let mut range = [0; Self::LEN + 1];
                let mut last_i = 0;
                for (i, r) in ranges.into_iter().enumerate() {
                    if i > Self::LEN {
                        break;
                    }
                    if i > 0 && range[i - 1] > r {
                        return Err(());
                    }
                    range[i] = r;
                    last_i = i;
                }
                if last_i != Self::LEN {
                    return Err(());
                }
                Ok(Self {
                    range,
                    bin: [0; Self::LEN],
                    stats: $crate::stats::OnlineStats::default(),
                    minmax: $crate::stats::MinMax::default(),
                })
            }

            /// Find the index of the bin corresponding to the given sample.
            ///
            /// Fails if the sample is out of range of the histogram.
            #[inline]
            pub fn find(&self, x: u64) -> Result<usize, ()> {
                // We made sure our ranges are valid at construction, so we can
                // safely unwrap.
                match self.range.binary_search_by(|p| p.partial_cmp(&x).unwrap()) {
                    Ok(i) if i < Self::LEN => {
                        Ok(i)
                    },
                    Err(i) if i > 0 && i < Self::LEN + 1 => {
                        Ok(i - 1)
                    },
                    _ => {
                        Err(())
                    },
                }
            }

            /// Return the bins of the histogram.
            #[inline]
            pub fn bins(&self) -> &[u64] {
                &self.bin[..]
            }

            /// Return the ranges of the histogram.
            #[inline]
            pub fn ranges(&self) -> &[u64] {
                &self.range[..]
            }

            /// Return an iterator over the bins and corresponding ranges:
            /// `((lower, upper), count)`
            #[inline]
            pub fn iter(&self) -> IterHistogram {
                self.into_iter()
            }

            /// Reset all bins to zero.
            #[inline]
            pub fn reset(&mut self) {
                self.bin = [0; Self::LEN];
            }

            /// Return the lower range limit.
            ///
            /// (The corresponding bin might be empty.)
            #[inline]
            pub fn range_min(&self) -> u64 {
                self.range[0]
            }

            /// Return the upper range limit.
            ///
            /// (The corresponding bin might be empty.)
            #[inline]
            pub fn range_max(&self) -> u64 {
                self.range[Self::LEN]
            }

            /// Return the minimum value observed so far
            #[inline]
            pub fn min(&self) -> u64 {
                self.minmax.min().copied().unwrap_or(0)
            }

            /// Return the maximum value observed so far
            #[inline]
            pub fn max(&self) -> u64 {
                self.minmax.max().copied().unwrap_or(0)
            }

            #[inline]
            pub fn mean(&self) -> f64 {
                self.stats.mean()
            }

            #[inline]
            pub fn stddev(&self) -> f64 {
                self.stats.stddev()
            }

            #[inline]
            pub fn variance(&self) -> f64 {
                self.stats.variance()
            }
        }

        impl $crate::stats::Histogram for Histogram {
            /// Add a sample to the histogram.
            ///
            /// Fails if the sample is out of range of the histogram.
            #[inline]
            fn add(&mut self, x: u64) -> Result<(), ()> {
                self.minmax.add(x);
                self.stats.add(x);
                if let Ok(i) = self.find(x) {
                    self.bin[i] += 1;
                    Ok(())
                } else {
                    Err(())
                }
            }
        }

        /// Iterate over all `(range, count)` pairs in the histogram.
        pub struct IterHistogram<'a> {
            remaining_bin: &'a [u64],
            remaining_range: &'a [u64],
        }

        impl<'a> core::iter::Iterator for IterHistogram<'a> {
            type Item = ((u64, u64), u64);
            fn next(&mut self) -> Option<((u64, u64), u64)> {
                if let Some((&bin, rest)) = self.remaining_bin.split_first() {
                    let left = self.remaining_range[0];
                    let right = self.remaining_range[1];
                    self.remaining_bin = rest;
                    self.remaining_range = &self.remaining_range[1..];
                    return Some(((left, right), bin));
                }
                None
            }
        }

        impl<'a> core::iter::IntoIterator for &'a Histogram {
            type Item = ((u64, u64), u64);
            type IntoIter = IterHistogram<'a>;
            fn into_iter(self) -> IterHistogram<'a> {
                IterHistogram {
                    remaining_bin: self.bins(),
                    remaining_range: self.ranges(),
                }
            }
        }

        impl core::fmt::Display for Histogram {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                use core::cmp;
                use core::fmt::Write;

                #[cfg(not(test))]
                use alloc::string::String;

                let num_samples: u64 = self.bins().iter().sum();
                writeln!(f, "# Number of samples = {}", num_samples)?;
                if num_samples == 0 {
                    return Ok(());
                }

                let min = self.minmax.min().unwrap();
                let max = self.minmax.max().unwrap();

                writeln!(f, "# Min = {}", min)?;
                writeln!(f, "# Max = {}", max)?;
                writeln!(f, "#")?;

                let mean = self.stats.mean();
                let dev = self.stats.stddev();
                let var = self.stats.variance();

                writeln!(f, "# Mean = {}", mean)?;
                writeln!(f, "# Standard deviation = {}", dev)?;
                writeln!(f, "# Variance = {}", var)?;
                writeln!(f, "#")?;

                let max_bucket_count = self.bins().iter().map(|b| *b).fold(0, cmp::max);

                const WIDTH: u64 = 50;
                let count_per_char = cmp::max(max_bucket_count / WIDTH, 1);

                writeln!(f, "# Each ∎ is a count of {}", count_per_char)?;
                writeln!(f, "#")?;

                let mut count_str = String::new();

                let widest_count = self.bins().iter().fold(0, |n, b| {
                    count_str.clear();
                    write!(&mut count_str, "{}", *b).unwrap();
                    cmp::max(n, count_str.len())
                });

                let mut end_str = String::new();
                let widest_range = self.ranges().iter().fold(0, |n, end| {
                    end_str.clear();
                    write!(&mut end_str, "{}", *end).unwrap();
                    cmp::max(n, end_str.len())
                });

                let mut start_str = String::with_capacity(widest_range);

                for ((start, end), bin) in self.into_iter() {
                    start_str.clear();
                    write!(&mut start_str, "{}", start).unwrap();
                    for _ in 0..widest_range - start_str.len() {
                        start_str.insert(0, ' ');
                    }

                    end_str.clear();
                    write!(&mut end_str, "{}", end).unwrap();
                    for _ in 0..widest_range - end_str.len() {
                        end_str.insert(0, ' ');
                    }

                    count_str.clear();
                    write!(&mut count_str, "{}", bin).unwrap();
                    for _ in 0..widest_count - count_str.len() {
                        count_str.insert(0, ' ');
                    }

                    write!(f, "{} .. {} [ {} ]: ", start_str, end_str, count_str)?;
                    for _ in 0..bin / count_per_char {
                        write!(f, "∎")?;
                    }
                    writeln!(f)?;
                }

                Ok(())
            }
        }
    };
}

mod defaults {
    // This histogram will spread all allocations across 100 buckets
    define_histogram!(100);
    // Provide a default implementation which will focus on the main sizes of concern
    impl Default for self::Histogram {
        fn default() -> Self {
            use heapless::Vec;
            use heapless::consts::U100;
            let mut ranges = Vec::<_, U100>::new();
            // Use the fibonnaci sequence up to 1TB
            let mut n: u64 = 1;
            let mut m: u64 = 2;
            ranges.push(m).unwrap();
            for _ in 1..57u64 {
                let new_m = n + m;
                n = m;
                ranges.push(new_m).unwrap();
                m = new_m;
            }
            // Grow by 20% afterwards
            for _ in 57..99u64 {
                let new_m = m + (m as f64 * 0.2).ceil() as u64;
                ranges.push(new_m).unwrap();
                m = new_m;
            }
            // Add one final range that covers the remaining address space
            ranges.push(u64::max_value()).unwrap();
            Self::from_ranges(ranges.iter().cloned()).unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    mod defaults {
        use crate::define_histogram;

        define_histogram!(10);
    }

    use super::Histogram;
    use self::defaults::Histogram as DefaultHistogram;

    #[test]
    fn histogram_with_const_width_test() {
        let mut h = DefaultHistogram::with_const_width(0, 100);
        for i in 0..100 {
            h.add(i).ok();
        }
        assert_eq!(h.max(), 99);
        assert_eq!(h.min(), 0);
        assert_eq!(h.mean(), 49.5);
        assert_eq!(h.bins(), &[10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    }

    #[test]
    fn histogram_from_ranges_test() {
        let ranges: [u64; 11] = [
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048
        ];
        let mut h = DefaultHistogram::from_ranges(ranges.iter().copied()).unwrap();
        for i in 2..2048 {
            h.add(i).ok();
        }
        assert_eq!(h.max(), 2047);
        assert_eq!(h.min(), 2);
        assert_eq!(h.mean(), 1024.5);
        assert_eq!(h.bins(), &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    }
}