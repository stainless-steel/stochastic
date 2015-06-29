//! Stochastic processes.

#[cfg(test)]
extern crate assert;

extern crate complex;
extern crate czt;
extern crate probability;

pub mod gaussian;

/// A stochastic process.
pub trait Process {
    /// The index set.
    type Index: Copy;

    /// The state space.
    type State;

    /// Compute the covariance.
    fn cov(&self, Self::Index, Self::Index) -> f64;

    /// Compute the variance.
    #[inline]
    fn var(&self, index: Self::Index) -> f64 {
        self.cov(index, index)
    }
}

/// A stationary process.
pub trait Stationary {
    /// The distance between two indices.
    type Distance: Distance;

    /// Compute the covariance.
    fn cov(&self, Self::Distance) -> f64;

    /// Compute the variance.
    #[inline]
    fn var(&self) -> f64 {
        self.cov(Self::Distance::zero())
    }
}

/// A distance.
pub trait Distance {
    /// The zero distance.
    fn zero() -> Self;
}

impl Distance for usize {
    #[inline(always)]
    fn zero() -> usize {
        0
    }
}
