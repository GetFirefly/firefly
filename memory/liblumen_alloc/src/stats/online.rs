#![allow(unused)]
///! Forked from github.com/BurntSushi/rust-stats
use core::default::Default;
use core::fmt;
use core::iter::{FromIterator, IntoIterator};

use num_traits::ToPrimitive;

use super::Commute;

/// Compute the standard deviation of a stream in constant space.
pub fn stddev<I>(it: I) -> f64
where
    I: Iterator,
    <I as Iterator>::Item: ToPrimitive,
{
    it.collect::<OnlineStats>().stddev()
}

/// Compute the variance of a stream in constant space.
pub fn variance<I>(it: I) -> f64
where
    I: Iterator,
    <I as Iterator>::Item: ToPrimitive,
{
    it.collect::<OnlineStats>().variance()
}

/// Compute the mean of a stream in constant space.
pub fn mean<I>(it: I) -> f64
where
    I: Iterator,
    <I as Iterator>::Item: ToPrimitive,
{
    it.collect::<OnlineStats>().mean()
}

/// Online state for computing mean, variance and standard deviation.
#[derive(Clone, Copy)]
pub struct OnlineStats {
    size: u64,
    mean: f64,
    variance: f64,
}

impl OnlineStats {
    /// Create initial state.
    ///
    /// Population size, variance and mean are set to `0`.
    pub fn new() -> OnlineStats {
        Default::default()
    }

    /// Initializes variance from a sample.
    pub fn from_slice<T: ToPrimitive>(samples: &[T]) -> OnlineStats {
        samples.iter().map(|n| n.to_f64().unwrap()).collect()
    }

    /// Return the current mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Return the current standard deviation.
    pub fn stddev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Return the current variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Add a new sample.
    pub fn add<T: ToPrimitive>(&mut self, sample: T) {
        let sample = sample.to_f64().unwrap();
        // Taken from: http://goo.gl/JKeqvj
        // See also: http://goo.gl/qTtI3V
        let oldmean = self.mean;
        let prevq = self.variance * (self.size as f64);

        self.size += 1;
        self.mean += (sample - oldmean) / (self.size as f64);
        self.variance = (prevq + (sample - oldmean) * (sample - self.mean)) / (self.size as f64);
    }

    /// Add a new NULL value to the population.
    ///
    /// This increases the population size by `1`.
    pub fn add_null(&mut self) {
        self.add(0usize);
    }

    /// Returns the number of data points.
    pub fn len(&self) -> usize {
        self.size as usize
    }
}

impl Commute for OnlineStats {
    fn merge(&mut self, v: OnlineStats) {
        // Taken from: http://goo.gl/iODi28
        let (s1, s2) = (self.size as f64, v.size as f64);
        let meandiffsq = (self.mean - v.mean) * (self.mean - v.mean);
        let mean = ((s1 * self.mean) + (s2 * v.mean)) / (s1 + s2);
        let var = (((s1 * self.variance) + (s2 * v.variance)) / (s1 + s2))
            + ((s1 * s2 * meandiffsq) / ((s1 + s2) * (s1 + s2)));
        self.size += v.size;
        self.mean = mean;
        self.variance = var;
    }
}

impl Default for OnlineStats {
    fn default() -> OnlineStats {
        OnlineStats {
            size: 0,
            mean: 0.0,
            variance: 0.0,
        }
    }
}

impl fmt::Debug for OnlineStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.10} +/- {:.10}", self.mean(), self.stddev())
    }
}

impl<T: ToPrimitive> FromIterator<T> for OnlineStats {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> OnlineStats {
        let mut v = OnlineStats::new();
        v.extend(it);
        v
    }
}

impl<T: ToPrimitive> Extend<T> for OnlineStats {
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        for sample in it {
            self.add(sample)
        }
    }
}

#[cfg(test)]
mod test {
    use super::OnlineStats;
    use crate::stats::{merge_all, Commute};

    #[test]
    fn stddev() {
        // TODO: Convert this to a quickcheck test.
        let expected = OnlineStats::from_slice(&[1usize, 2, 3, 2, 4, 6]);

        let var1 = OnlineStats::from_slice(&[1usize, 2, 3]);
        let var2 = OnlineStats::from_slice(&[2usize, 4, 6]);
        let mut got = var1;
        got.merge(var2);
        assert_eq!(expected.stddev(), got.stddev());
    }

    #[test]
    fn stddev_many() {
        // TODO: Convert this to a quickcheck test.
        let expected = OnlineStats::from_slice(&[1usize, 2, 3, 2, 4, 6, 3, 6, 9]);

        let vars = vec![
            OnlineStats::from_slice(&[1usize, 2, 3]),
            OnlineStats::from_slice(&[2usize, 4, 6]),
            OnlineStats::from_slice(&[3usize, 6, 9]),
        ];
        assert_eq!(
            expected.stddev(),
            merge_all(vars.into_iter()).unwrap().stddev()
        );
    }
}
