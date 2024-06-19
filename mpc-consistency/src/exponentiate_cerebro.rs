use std::ops::Mul;
use ark_bls12_377::g1::Config;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::{AffineRepr, Group, ScalarMul, VariableBaseMSM};
use ark_ec::bls12::Bls12;
use ark_ec::bn::G1Affine;
use ark_ec::pairing::Pairing;
use ark_ec::short_weierstrass::Projective;
use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_std::{cfg_into_iter, cfg_iter, start_timer, UniformRand};
use blake2::Blake2s256;
use log::debug;
use num_traits::One;
use num_traits::real::Real;
use crate::ped_pc::BaselinePedPC;
use structopt::StructOpt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod prove_verify;
mod gen_pp;
mod common;
mod mpspdz;
mod ped_pc;
mod perf_trace_structured;

pub type E = <Bls12<ark_bls12_377::Config> as Pairing>::G1Affine;
pub type F = <E as AffineRepr>::ScalarField;
pub type UniPoly_377 = DensePolynomial<F>;
pub type Sponge_Bls12_377 = PoseidonSponge<<E as AffineRepr>::ScalarField>;

// pub type PCS = InnerProductArgPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377>;
pub type PCS = BaselinePedPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377>;

#[derive(Debug, StructOpt)]
#[structopt(name = "bench", about = "BenchCommit")]
struct Opt {
    /// Activate debug mode
    // short and long flags (-d, --debug) will be deduced from the field's name
    #[structopt(short, long)]
    debug: bool,

    /// Which party are you? 0 or 1?
    #[structopt(long, default_value = "10")]
    n_parameters: u64,

}

fn main() {
    debug!("Generating public parameters and exchanging commitments");
    let opt = Opt::from_args();
    if opt.debug {
        env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::init();
    }

    let n_parameters = opt.n_parameters;
    let mut rng = &mut ark_std::test_rng();
    let generator = E::generator().into_group();

    let param_range = (0..n_parameters).collect::<Vec<_>>();
    let commitment_rands: Vec<F> = param_range.into_iter()
        .map(|_| F::rand(&mut rng))
        .collect();

    let commitments = cfg_into_iter!(commitment_rands)
        .map(|r| generator.mul(r))
        .collect::<Vec<_>>();

    let t_exponentiate = start_timer!(|| "Exponentiate");

    let t_batch_convert = start_timer!(|| "Batch convert");
    let commitments_batch = Projective::<Config>::batch_convert_to_mul_base(commitments.as_slice());
    end_timer!(t_batch_convert);

    let point = F::rand(&mut rng);
    let mut beta: F = F::one();

    let mut scalars_bigint: Vec<<F as PrimeField>::BigInt> = vec![];
    for _ in 0..n_parameters {
        scalars_bigint.push(beta.into_bigint());
        beta = beta.mul(&point);
    }
    let t_msm = start_timer!(|| "MSM");
    let rho_com = <Projective<Config> as VariableBaseMSM>::msm_bigint(&commitments_batch, &scalars_bigint);
    end_timer!(t_msm);
    println!("rho_com: {:?} {:?} {:?}", rho_com.x.to_string(), rho_com.y.to_string(), rho_com.z.to_string());

    end_timer!(t_exponentiate);
}
