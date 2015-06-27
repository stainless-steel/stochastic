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
    type Index;

    /// The state space.
    type State;

    /// Sample paths.
    type Path: Path<Self::State>;

    /// Compute the covariance function.
    fn cov(&self, Self::Index, Self::Index) -> f64;
}

/// A sample path of a stochastic process.
pub trait Path<T>: Iterator<Item=T> {
}

/// A stationary process.
pub trait Stationary {
    /// The index set.
    type Index;

    /// Compute the covariance functino.
    fn cov(&self, Self::Index) -> f64;
}
