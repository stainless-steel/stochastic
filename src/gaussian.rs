//! Gaussian processes.

use complex::{Complex, c64};
use probability::distribution::{Distribution, Gaussian};
use probability::generator::Generator;

use {Path, Process, Stationary};

/// A fractional Gaussian noise.
pub struct FractionalNoise {
    hurst: f64,
}

/// A sample path of a fractional Gaussian noise.
pub struct FractionalNoisePath {
    position: usize,
    data: Vec<f64>,
}

impl FractionalNoise {
    /// Create a fractional Gaussian noise.
    #[inline]
    pub fn new(hurst: f64) -> FractionalNoise {
        debug_assert!(hurst > 0.0 && hurst < 1.0);
        FractionalNoise { hurst: hurst }
    }

    /// Generate a sample path.
    #[inline]
    pub fn sample<G>(&self, size: usize, generator: &mut G) -> FractionalNoisePath
        where G: Generator
    {
        FractionalNoisePath::new(self, size, generator)
    }
}

impl Stationary for FractionalNoise {
    type Index = usize;

    fn cov(&self, tau: usize) -> f64 {
        let tau = tau as f64;
        let power = 2.0 * self.hurst;
        0.5 * ((tau + 1.0).powf(power) - 2.0 * tau.powf(power) + (tau - 1.0).abs().powf(power))
    }
}

impl FractionalNoisePath {
    #[inline]
    fn new<G>(noise: &FractionalNoise, size: usize, generator: &mut G) -> FractionalNoisePath
        where G: Generator
    {
        FractionalNoisePath {
            position: 0,
            data: {
                let gaussian = Gaussian::new(0.0, 1.0);
                let scale = (1.0 / (size - 1) as f64).powf(noise.hurst);
                let data = circulant_embedding(noise, size, || gaussian.sample(generator));
                data.iter().take(size).map(|point| scale * point.re()).collect()
            },
        }
    }
}

impl Process for FractionalNoise {
    type Index = usize;
    type State = f64;
    type Path = FractionalNoisePath;

    #[inline]
    fn cov(&self, t: usize, s: usize) -> f64 {
        Stationary::cov(self, if t < s { s - t } else { t - s })
    }
}

impl Path<f64> for FractionalNoisePath {
}

impl Iterator for FractionalNoisePath {
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

/// Compute two independent sample paths stored in the real and complex parts of
/// a sequence of `2 × n` complex numbers.
///
/// References:
///
/// 1. Dirk P. Kroese, Thomas Taimre, and Zdravko I. Botev. Handbook for Monte
///    Carlo Methods. Hoboken, N.J.: Wiley, 2011.
///
/// 2. C. R. Dietrich and G. N. Newsam. “Fast and Exact Simulation of Stationary
///    Gaussian Processes Through Circulant Embedding of the Covariance Matrix.”
///    Siam Journal on Scientific Computing, 1997.
fn circulant_embedding<P, F>(process: &P, n: usize, mut gaussian: F) -> Vec<c64>
    where P: Process<Index=usize, State=f64> + Stationary<Index=usize>, F: FnMut() -> f64
{
    use czt;

    macro_rules! chirp(
        ($m:expr) => ({
            use std::f64::consts::PI;
            c64::from_polar(1.0, -2.0 * PI / $m as f64)
        });
    );

    let m = (1 + n) + (1 + n) - 2;
    let mut data = vec![0.0; m];
    {
        data[0] = Stationary::cov(process, 0);
        for i in 1..(n + 1) {
            data[i] = Stationary::cov(process, i);
            data[m - i] = data[i];
        }
    }

    let mut data = czt::forward(&data, m, chirp!(m), c64(1.0, 0.0));
    {
        let scale = 1.0 / (2 * n) as f64;
        for i in 0..m {
            if cfg!(debug_assertions) {
                const EPSILON: f64 = 1e-10;
                assert!(data[i].re() > -EPSILON);
                assert!(data[i].im().abs() < EPSILON);
            }
            let sigma = (data[i].re().max(0.0) * scale).sqrt();
            data[i] = c64(sigma * gaussian(), sigma * gaussian());
        }
    }

    czt::forward(&mut data, m, chirp!(m), c64(1.0, 0.0))
}

#[cfg(test)]
mod tests {
    use assert;
    use complex::Complex;
    use gaussian::FractionalNoise;

    #[test]
    fn circulant_embedding() {
        let gaussians = [
             5.376671395461000e-01, -8.044659563495471e-01,
             1.833885014595086e+00,  6.966244158496073e-01,
            -2.258846861003648e+00,  8.350881650726819e-01,
             8.621733203681206e-01, -2.437151403779522e-01,
             3.187652398589808e-01,  2.156700864037444e-01,
            -1.307688296305273e+00, -1.165843931482049e+00,
            -4.335920223056836e-01, -1.147952778898594e+00,
             3.426244665386499e-01,  1.048747160164940e-01,
             3.578396939725760e+00,  7.222540322250016e-01,
             2.769437029884877e+00,  2.585491252616241e+00,
            -1.349886940156521e+00, -6.668906707013855e-01,
             3.034923466331855e+00,  1.873310245789398e-01,
             7.254042249461056e-01, -8.249442537095540e-02,
            -6.305487318965619e-02, -1.933022917850987e+00,
             7.147429038260958e-01, -4.389661539347733e-01,
            -2.049660582997746e-01, -1.794678841455123e+00,
            -1.241443482163119e-01,  8.403755297539054e-01,
             1.489697607785465e+00, -8.880320823290103e-01,
             1.409034489800479e+00,  1.000928331393225e-01,
             1.417192413429614e+00, -5.445289299905477e-01,
             6.714971336080805e-01,  3.035207946493543e-01,
            -1.207486922685038e+00, -6.003265621337341e-01,
             7.172386513288385e-01,  4.899653211739479e-01,
             1.630235289164729e+00,  7.393631236044740e-01,
             4.888937703117894e-01,  1.711887782981555e+00,
             1.034693009917860e+00, -1.941235357582654e-01,
             7.268851333832379e-01, -2.138355269439939e+00,
            -3.034409247860159e-01, -8.395887473366136e-01,
             2.938714670966581e-01,  1.354594328004644e+00,
            -7.872828037586376e-01, -1.072155288384252e+00,
             8.883956317576418e-01,  9.609538697405668e-01,
            -1.147070106969150e+00,  1.240498000031925e-01,
            -1.068870458168032e+00,  1.436696622718939e+00,
            -8.094986944248755e-01, -1.960899999365033e+00,
            -2.944284161994896e+00, -1.976982259741502e-01,
             1.438380292815098e+00, -1.207845485259799e+00,
             3.251905394561979e-01,  2.908008030729362e+00,
            -7.549283191697034e-01,  8.252188942284909e-01,
             1.370298540095228e+00,  1.378971977916614e+00,
            -1.711516418853698e+00, -1.058180257987362e+00,
            -1.022424460854909e-01, -4.686155811006241e-01,
            -2.414470416073579e-01, -2.724694092501875e-01,
             3.192067391655018e-01,  1.098424617888623e+00,
             3.128585966374284e-01, -2.778719327876389e-01,
            -8.648799173244565e-01,  7.015414581632835e-01,
            -3.005129619626856e-02, -2.051816299911149e+00,
            -1.648790192090383e-01, -3.538499977744333e-01,
             6.277072875287265e-01, -8.235865251568527e-01,
             1.093265669039484e+00, -1.577057022799202e+00,
             1.109273297614398e+00,  5.079746509059456e-01,
            -8.636528219887144e-01,  2.819840636705562e-01,
             7.735909113042493e-02,  3.347988224445142e-02,
            -1.214117043615409e+00, -1.333677943428106e+00,
            -1.113500741486764e+00,  1.127492278341590e+00,
            -6.849328103348064e-03,  3.501794106033116e-01,
             1.532630308284750e+00, -2.990660303329824e-01,
            -7.696659137536819e-01,  2.288979275162977e-02,
             3.713788127600577e-01, -2.619954349660920e-01,
            -2.255844022712519e-01, -1.750212368446790e+00,
             1.117356138814467e+00, -2.856509715953298e-01,
            -1.089064295052236e+00, -8.313665115676243e-01,
             3.255746416497347e-02, -9.792063051673021e-01,
             5.525270211122237e-01, -1.156401655664002e+00,
             1.100610217880866e+00, -5.335571093159874e-01,
             1.544211895503951e+00, -2.002635735883060e+00,
             8.593113317542546e-02,  9.642294226316275e-01,
            -1.491590310637609e+00,  5.200601014554576e-01,
            -7.423018372598567e-01, -2.002785164253808e-02,
            -1.061581733319986e+00, -3.477108602848296e-02,
             2.350457224002042e+00, -7.981635845641424e-01,
            -6.156018814668943e-01,  1.018685282128575e+00,
             7.480767837039851e-01, -1.332174795077347e-01,
            -1.924185105882636e-01, -7.145301637871584e-01,
             8.886104254207208e-01,  1.351385768426657e+00,
            -7.648492365678742e-01, -2.247710560525841e-01,
            -1.402268969338759e+00, -5.890290307208013e-01,
            -1.422375925091496e+00, -2.937535977354161e-01,
             4.881939098599407e-01, -8.479262436379339e-01,
            -1.773751566188252e-01, -1.120128301243728e+00,
            -1.960534878073328e-01,  2.525999692118309e+00,
             1.419310150642549e+00,  1.655497592887346e+00,
             2.915843739841825e-01,  3.075351592382519e-01,
             1.978110534643607e-01, -1.257118359352053e+00,
             1.587699089974059e+00, -8.654680305548038e-01,
        ];

        let expected_data = [
             5.342797518772909e-01,
             1.369742853760759e+00,
             4.111771343211609e-01,
             2.294009764716167e-01,
             6.162829589777247e-01,
             9.007956361348235e-01,
             7.023558622059536e-02,
             3.116111226165416e-01,
            -6.002256180770911e-02,
             5.012033271784876e-01,
             3.150108647409661e-01,
             9.428387384854191e-02,
             1.607312734184252e+00,
             9.372040952059283e-01,
             1.981144400750289e+00,
             1.282881390320189e+00,
             1.563732961490111e+00,
             1.748856195530649e+00,
             1.365235515586270e+00,
             1.601191884438295e+00,
             1.096521319802640e+00,
             1.312852711461883e+00,
             2.306356423335515e+00,
             2.558697847363409e+00,
             2.603675844300237e+00,
             1.862884079689471e+00,
             2.189904103639442e+00,
             1.681372459404732e+00,
             1.520614417799068e+00,
             2.229707785601377e+00,
             2.283669380913793e+00,
             3.189222945083247e+00,
             2.002002529264580e+00,
             2.327053362601621e+00,
             1.909998920631301e+00,
             1.606864287044382e+00,
             1.045465894314694e+00,
             7.083020094566839e-01,
             1.397354877258026e+00,
             1.126068340196503e+00,
             1.975245348794911e+00,
             8.402934102746060e-01,
             1.004117183703313e-01,
        ];

        let mut k = 0;
        let gaussian = || { k += 1; gaussians[k - 1] };

        let hurst = 0.2;
        let process = FractionalNoise::new(hurst);

        let n = 42;
        let data = super::circulant_embedding(&process, n, gaussian);

        let mut sum = 0.0;
        let scale = (n as f64).powf(-hurst);
        let data = data.iter().take(n + 1).map(|point| {
            sum += scale * point.re();
            sum
        }).collect::<Vec<_>>();

        assert::close(&data, &expected_data[..], 1e-13);
    }
}
