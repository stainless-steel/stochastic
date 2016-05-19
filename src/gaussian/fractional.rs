//! Fractional Brownian motion and fractional Gaussian noise.

use probability::distribution::{Gaussian, Sample};
use probability::source::Source;

use {Process, Stationary};
use gaussian::circulant_embedding;

macro_rules! hurst(
    ($value:expr) => ({
        let value = $value;
        debug_assert!(value > 0.0 && value < 1.0);
        value
    });
);

macro_rules! step(
    ($value:expr) => ({
        let value = $value;
        debug_assert!(value > 0.0);
        value
    });
);

/// A fractional Brownian motion.
pub struct Motion {
    hurst: f64,
}

/// A fractional Gaussian noise.
pub struct Noise {
    hurst: f64,
    step: f64,
}

impl Motion {
    /// Create a fractional Brownian motion.
    #[inline]
    pub fn new(hurst: f64) -> Motion {
        Motion { hurst: hurst!(hurst) }
    }

    /// Generate a sample path.
    pub fn sample<S>(&self, points: usize, step: f64, source: &mut S) -> Vec<f64>
        where S: Source
    {
        match points {
            0 => vec![],
            1 => vec![0.0],
            _ => {
                let mut data = vec![0.0];
                data.extend(Noise::new(self.hurst, step).sample(points - 1, source));
                for i in 2..points {
                    data[i] += data[i - 1];
                }
                data
            },
        }
    }
}

impl Noise {
    /// Create a fractional Gaussian noise.
    #[inline]
    pub fn new(hurst: f64, step: f64) -> Noise {
        Noise { hurst: hurst!(hurst), step: step!(step) }
    }

    /// Generate a sample path.
    pub fn sample<S>(&self, points: usize, source: &mut S) -> Vec<f64>
        where S: Source
    {
        match points {
            0 => vec![],
            1 => vec![Gaussian::new(0.0, Stationary::var(self).sqrt()).sample(source)],
            _ => {
                let n = points - 1;
                let gaussian = Gaussian::new(0.0, 1.0);
                let data = circulant_embedding(self, n, || gaussian.sample(source));
                data.iter().take(points).map(|point| point.re).collect()
            },
        }
    }
}

impl Process for Motion {
    type Index = f64;
    type State = f64;

    fn cov(&self, t: f64, s: f64) -> f64 {
        debug_assert!(t >= 0.0 && s >= 0.0);
        let power = 2.0 * self.hurst;
        0.5 * (t.powf(power) + s.powf(power) - (t - s).abs().powf(power))
    }
}

impl Process for Noise {
    type Index = usize;
    type State = f64;

    #[inline]
    fn cov(&self, t: usize, s: usize) -> f64 {
        Stationary::cov(self, if t < s { s - t } else { t - s })
    }
}

impl Stationary for Noise {
    type Distance = usize;

    fn cov(&self, delta: usize) -> f64 {
        let delta = delta as f64;
        let power = 2.0 * self.hurst;
        let var = self.step.powf(power);
        0.5 * var * ((delta + 1.0).powf(power) - 2.0 * delta.powf(power) +
                     (delta - 1.0).abs().powf(power))
    }
}

#[cfg(test)]
mod tests {
    use super::{Motion, Noise};

    #[test]
    fn motion_var() {
        use Process;
        let process = Motion::new(0.25);
        let variances = (0..3).map(|i| process.var(i as f64)).collect::<Vec<_>>();
        assert_eq!(&variances, &[0.0, 1.0, 2f64.sqrt()]);
    }

    #[test]
    fn noise_var() {
        let process = Noise::new(0.42, 1.0);
        {
            use Process;
            let variances = (0..3).map(|i| process.var(i)).collect::<Vec<_>>();
            assert_eq!(&variances, &[1.0, 1.0, 1.0]);
        }
        {
            use Stationary;
            assert_eq!(process.var(), 1.0);
        }
        let process = Noise::new(0.25, 0.01);
        {
            use Process;
            let variances = (0..3).map(|i| process.var(i)).collect::<Vec<_>>();
            assert_eq!(&variances, &[0.1, 0.1, 0.1]);
        }
        {
            use Stationary;
            assert_eq!(process.var(), 0.1);
        }
    }
}
