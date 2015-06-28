use complex::Complex;
use probability::distribution::{Distribution, Gaussian};
use probability::generator::Generator;

use {Path, Process, Stationary, gaussian};

/// A fractional Gaussian noise.
pub struct Noise {
    hurst: f64,
}

/// A sample path of a fractional Gaussian noise.
pub struct NoisePath {
    position: usize,
    data: Vec<f64>,
}

impl Noise {
    /// Create a fractional Gaussian noise.
    #[inline]
    pub fn new(hurst: f64) -> Noise {
        Noise { hurst: hurst!(hurst) }
    }

    /// Generate a sample path.
    #[inline]
    pub fn sample<G>(&self, count: usize, generator: &mut G) -> NoisePath
        where G: Generator
    {
        NoisePath::new(self, count, generator)
    }
}

impl Process for Noise {
    type Index = usize;
    type State = f64;
    type Path = NoisePath;

    #[inline]
    fn cov(&self, t: usize, s: usize) -> f64 {
        Stationary::cov(self, if t < s { s - t } else { t - s })
    }
}

impl Stationary for Noise {
    type Index = usize;

    fn cov(&self, tau: usize) -> f64 {
        let tau = tau as f64;
        let power = 2.0 * self.hurst;
        0.5 * ((tau + 1.0).powf(power) - 2.0 * tau.powf(power) + (tau - 1.0).abs().powf(power))
    }
}

impl NoisePath {
    #[inline]
    fn new<G>(noise: &Noise, count: usize, generator: &mut G) -> NoisePath
        where G: Generator
    {
        let data = match count {
            0 => vec![],
            1 => vec![Gaussian::new(0.0, 1.0).sample(generator)],
            _ => {
                let n = count - 1;
                let gaussian = Gaussian::new(0.0, 1.0);
                let scale = (1.0 / n as f64).powf(noise.hurst);
                let data = gaussian::circulant_embedding(noise, n, || gaussian.sample(generator));
                data.iter().take(count).map(|point| scale * point.re()).collect()
            },
        };
        NoisePath { position: 0, data: data }
    }

    #[inline]
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }
}

impl Path<f64> for NoisePath {
}

impl Iterator for NoisePath {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            None
        } else {
            let state = self.data[self.position];
            self.position += 1;
            Some(state)
        }
    }
}
