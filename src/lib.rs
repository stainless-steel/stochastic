//! Stochastic processes.

extern crate fft;
extern crate probability;

pub mod gaussian;

/// A stochastic process.
pub trait Process {
    /// States.
    type State;

    /// Sample paths.
    type Trace: Trace<Self::State>;
}

/// A sample path of a stochastic process.
pub trait Trace<T>: Iterator<Item=T> {
}
