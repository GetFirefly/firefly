//! This library is a fork of the `histo` crate, which is dual MIT/Apache 2 licensed
//! software, and can be found [here](https://github.com/fitzgen/histo).
//!
//! This was forked to make it `no_std` compatible, and to have finer control over its
//! performance characteristics since this is used in a very hot path (allocations).
//!
//! # Documentation
//!
//! Histograms with a configurable number of buckets, and a terminal-friendly
//! `Display`.
//!
//! This crate provides a `Histogram` type that allows configuration of the
//! number of buckets that will be used, regardless of the range of input
//! samples. This is useful when displaying a `Histogram` (for example, when
//! printing it to a terminal) but it sacrifices fancy tracking of precision and
//! significant figures.
//!
//! It uses O(n) memory.
//!
//! ```
//! use liblumen_alloc::stats::Histogram;
//!
//! # fn main() {
//! // Create a histogram that will have 10 buckets.
//! let mut histogram = Histogram::with_buckets(10);
//!
//! // Adds some samples to the histogram.
//! for sample in 0..100 {
//!     histogram.add(sample);
//!     histogram.add(sample * sample);
//! }
//!
//! // Iterate over buckets and do stuff with their range and count.
//! for bucket in histogram.buckets() {
//! #   let do_stuff = |_, _, _| {};
//!     do_stuff(bucket.start(), bucket.end(), bucket.count());
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
//!
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

#[cfg(all(test, feature = "quickcheck"))]
#[macro_use]
extern crate quickcheck;

use core::cmp;
use core::fmt;

use alloc::collections::btree_map::Range;
use alloc::collections::BTreeMap;

use super::{MinMax, OnlineStats};

/// A histogram is a collection of samples, sorted into buckets.
///
/// See the crate level documentation for more details.
#[derive(Debug, Clone)]
pub struct Histogram {
    num_buckets: u64,
    samples: BTreeMap<u64, u64>,
    stats: OnlineStats,
    minmax: MinMax<u64>,
}

impl Histogram {
    /// Construct a new histogram with the given number of buckets.
    ///
    /// ## Panics
    ///
    /// Panics if the number of buckets is zero.
    pub fn with_buckets(num_buckets: u64) -> Histogram {
        assert!(num_buckets > 0);
        Histogram {
            num_buckets,
            samples: Default::default(),
            stats: Default::default(),
            minmax: Default::default(),
        }
    }

    /// Add a new sample to this histogram.
    pub fn add(&mut self, sample: u64) {
        *self.samples.entry(sample).or_insert(0) += 1;
        self.minmax.add(sample);
        self.stats.add(sample);
    }

    /// Get an iterator over this histogram's buckets.
    pub fn buckets(&self) -> Buckets {
        Buckets {
            histogram: self,
            index: 0,
        }
    }
}

impl fmt::Display for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use alloc::string::String;
        use core::fmt::Write;

        let num_samples: u64 = self.samples.values().sum();
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

        let max_bucket_count = self.buckets().map(|b| b.count()).fold(0, cmp::max);

        const WIDTH: u64 = 50;
        let count_per_char = cmp::max(max_bucket_count / WIDTH, 1);

        writeln!(f, "# Each ∎ is a count of {}", count_per_char)?;
        writeln!(f, "#")?;

        let mut count_str = String::new();

        let widest_count = self.buckets().fold(0, |n, b| {
            count_str.clear();
            write!(&mut count_str, "{}", b.count()).unwrap();
            cmp::max(n, count_str.len())
        });

        let mut end_str = String::new();
        let widest_range = self.buckets().fold(0, |n, b| {
            end_str.clear();
            write!(&mut end_str, "{}", b.end()).unwrap();
            cmp::max(n, end_str.len())
        });

        let mut start_str = String::with_capacity(widest_range);

        for bucket in self.buckets() {
            start_str.clear();
            write!(&mut start_str, "{}", bucket.start()).unwrap();
            for _ in 0..widest_range - start_str.len() {
                start_str.insert(0, ' ');
            }

            end_str.clear();
            write!(&mut end_str, "{}", bucket.end()).unwrap();
            for _ in 0..widest_range - end_str.len() {
                end_str.insert(0, ' ');
            }

            count_str.clear();
            write!(&mut count_str, "{}", bucket.count()).unwrap();
            for _ in 0..widest_count - count_str.len() {
                count_str.insert(0, ' ');
            }

            write!(f, "{} .. {} [ {} ]: ", start_str, end_str, count_str)?;
            for _ in 0..bucket.count() / count_per_char {
                write!(f, "∎")?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// An iterator over the buckets in a histogram.
#[derive(Debug, Clone)]
pub struct Buckets<'a> {
    histogram: &'a Histogram,
    index: u64,
}

impl<'a> Iterator for Buckets<'a> {
    type Item = Bucket<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.histogram.num_buckets {
            return None;
        }

        let (min, max) = match (self.histogram.minmax.min(), self.histogram.minmax.max()) {
            (Some(&min), Some(&max)) => (min, max),
            _ => return None,
        };

        let range = max - min;
        let range = range + (range % self.histogram.num_buckets);

        let bucket_size = range / self.histogram.num_buckets;
        let bucket_size = cmp::max(1, bucket_size);

        let start = min + self.index * bucket_size;
        let end = min + (self.index + 1) * bucket_size;

        self.index += 1;

        Some(Bucket {
            start,
            end,
            range: if self.index == self.histogram.num_buckets {
                self.histogram.samples.range(start..)
            } else {
                self.histogram.samples.range(start..end)
            },
        })
    }
}

/// A bucket is a range of samples and their count.
#[derive(Clone)]
pub struct Bucket<'a> {
    start: u64,
    end: u64,
    range: Range<'a, u64, u64>,
}

impl<'a> Bucket<'a> {
    /// The number of samples in this bucket's range.
    pub fn count(&self) -> u64 {
        self.range.clone().map(|(_, count)| count).sum()
    }

    /// The start of this bucket's range.
    pub fn start(&self) -> u64 {
        self.start
    }

    /// The end of this bucket's range.
    pub fn end(&self) -> u64 {
        self.end
    }
}

impl<'a> fmt::Debug for Bucket<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bucket {{ {}..{} }}", self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_buckets() {
        let mut histo = Histogram::with_buckets(10);
        for i in 0..100 {
            histo.add(i);
        }
        assert_eq!(histo.buckets().count(), 10);
    }

    #[test]
    fn bucket_count() {
        let mut histo = Histogram::with_buckets(1);
        for i in 0..10 {
            histo.add(i);
        }
        assert_eq!(histo.buckets().count(), 1);
        assert_eq!(histo.buckets().next().unwrap().count(), 10);
    }

    #[test]
    fn bucket_count_one() {
        let mut histo = Histogram::with_buckets(1);
        histo.add(1);
        assert_eq!(histo.buckets().count(), 1);
        assert_eq!(histo.buckets().next().unwrap().count(), 1);
    }

    #[test]
    fn overflow_panic() {
        use alloc::string::ToString;

        let mut histo = Histogram::with_buckets(1);
        histo.add(99);
        histo.to_string();
    }
}

#[cfg(all(test, feature = "quickcheck"))]
mod quickchecks {
    use super::*;
    use core::cmp;

    quickcheck! {
        fn sum_of_bucket_counts_is_total_count(buckets: u64, samples: Vec<u64>) -> () {
            if buckets == 0 {
                return;
            }

            let len = samples.len();
            let mut histo = Histogram::with_buckets(buckets);
            for s in samples {
                histo.add(s);
            }

            assert_eq!(len, histo.stats.len(), "stats.len() should be correct");
            assert_eq!(len, histo.minmax.len(), "minmax.len() should be correct");
            assert_eq!(len as u64,
                       histo.samples.values().cloned().sum::<u64>(),
                       "samples.values() count should be correct");
            assert_eq!(len as u64,
                       histo.buckets().map(|b| b.count()).sum::<u64>(),
                       "sum of buckets counts should be correct");
        }

        fn actual_buckets_should_be_less_than_or_equal_num_buckets(
            buckets: u64,
            samples: Vec<u64>
        ) -> () {
            if buckets == 0 {
                return;
            }

            let mut histo = Histogram::with_buckets(buckets);
            for s in samples {
                histo.add(s);
            }

            assert!(histo.buckets().count() as u64 <= buckets,
                    "should never have more than expected number of buckets");
        }

        fn bucket_ranges_should_be_correct(buckets: u64, samples: Vec<u64>) -> () {
            if buckets == 0 {
                return;
            }

            let mut histo = Histogram::with_buckets(buckets);
            for s in samples {
                histo.add(s);
            }

            histo.buckets()
                .fold(None, |minmax, bucket| {
                    let bucket_range = bucket.end() - bucket.start();
                    let (min, max) = minmax.unwrap_or((bucket_range, bucket_range));
                    let min = cmp::min(min, bucket_range);
                    let max = cmp::max(max, bucket_range);
                    assert!(max - min <= 1);
                    Some((min, max))
                });
        }

        fn formatting_should_never_panic(buckets: u64, samples: Vec<u64>) -> () {
            use alloc::string::ToString;

            if buckets == 0 {
                return;
            }

            let mut histo = Histogram::with_buckets(buckets);
            for s in samples {
                histo.add(s);
            }

            histo.to_string();
        }
    }
}
