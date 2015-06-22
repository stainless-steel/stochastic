//! Gaussian processes.

use probability::Generator;

use {Process, Trace};

/// A fractional Gaussian noise.
pub struct FractionalNoise {
    hurst: f64,
}

/// A sample path of a fractional Gaussian noise.
pub struct FractionalNoiseTrace {
    position: usize,
    data: Vec<f64>,
}

impl FractionalNoise {
    /// Create a fractional Gaussian noise.
    #[inline]
    pub fn new(hurst: f64) -> FractionalNoise {
        debug_assert!(hurst > 0.0 && hurst < 1.0, "the Hurst parameter should be in (0, 1)");
        FractionalNoise { hurst: hurst }
    }

    /// Generate a sample path.
    #[inline]
    pub fn sample<G>(&self, size: usize, generator: &mut G) -> FractionalNoiseTrace
        where G: Generator
    {
        FractionalNoiseTrace::new(self, size, generator)
    }
}

impl FractionalNoiseTrace {
    fn new<G>(noise: &FractionalNoise, size: usize, generator: &mut G) -> FractionalNoiseTrace
        where G: Generator
    {
        FractionalNoiseTrace {
            position: 0,
            data: (0..size).map(|_| {
                noise.hurst * generator.gen::<f64>()
            }).collect(),
        }
    }
}

impl Process for FractionalNoise {
    type State = f64;
    type Trace = FractionalNoiseTrace;
}

impl Trace<f64> for FractionalNoiseTrace {
}

impl Iterator for FractionalNoiseTrace {
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

#[cfg(test)]
mod tests {
    use gaussian::FractionalNoise;
    use probability::generator;

    #[test]
    fn sample() {
        let noise = FractionalNoise::new(0.5);
        for _ in noise.sample(10, &mut generator()) {
        }
    }
}
