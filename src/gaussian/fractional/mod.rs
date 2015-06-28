//! Fractional Brownian motion and Gaussian noise.

macro_rules! hurst(
    ($hurst:expr) => ({
        let hurst = $hurst;
        debug_assert!(hurst > 0.0 && hurst < 1.0);
        hurst
    });
);

mod noise;

pub use self::noise::{Noise, NoisePath};
