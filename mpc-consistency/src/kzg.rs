

use ark_poly_commit::{Polynomial, marlin_pc::MarlinKZG10, LabeledPolynomial, PolynomialCommitment, QuerySet, Evaluations, challenge::ChallengeGenerator};
use ark_bls12_377::{Fr, Bls12_377};
use ark_crypto_primitives::sponge::poseidon::{PoseidonSponge, PoseidonConfig};
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;

use ark_ff::{Field, PrimeField, Fp256, BigInteger256, BigInteger};
use ark_poly::domain::radix2::Radix2EvaluationDomain;
use ark_poly::univariate::{DensePolynomial};
// use ark_poly::{EvaluationDomain, Polynomial};
use ark_poly_commit::{marlin_pc, DenseUVPolynomial};
use ark_serialize::CanonicalSerialize;
// use ark_std::rand::SeedableRng;
use std::borrow::Cow;
use std::path::PathBuf;
use ark_poly_commit::kzg10::{Commitment, Randomness};
use ark_poly_commit::sonic_pc::UniversalParams;
use clap::{ValueEnum};
use structopt::StructOpt;
use rand::{SeedableRng, Rng, rngs::StdRng};

use ark_std::{start_timer, end_timer, test_rng};
use log::debug;

type E = Bls12_377;
type F = <E as Pairing>::ScalarField;
type UniPoly_377 = DensePolynomial<F>;



pub fn kzg_setup(n_args: usize, mut rng: StdRng) -> UniversalParams<Bls12_377> {
    return ark_poly_commit::kzg10::KZG10::<
        Bls12_377,
        DensePolynomial<<Bls12<ark_bls12_377::Config> as Pairing>::ScalarField>,
    >::setup(n_args - 1, false, &mut rng)
        .unwrap();
}

pub fn kzg_commit(pp: &UniversalParams<Bls12_377>, inputs: &Vec<F>) -> (Commitment<Bls12_377>, Randomness<Fr, DensePolynomial<Fr>>) {

    let poly = UniPoly_377::from_coefficients_slice(&inputs[..]);
    let powers_of_gamma_g = (0..inputs.len())
        .map(|i| pp.powers_of_gamma_g[&i])
        .collect::<Vec<_>>();
    let powers = ark_poly_commit::kzg10::Powers::<Bls12_377> {
        powers_of_g: Cow::Borrowed(&pp.powers_of_g),
        powers_of_gamma_g: Cow::Owned(powers_of_gamma_g),
    };
    // IS this a non-pedersen version?
    let (commit, rand) =
        ark_poly_commit::kzg10::KZG10::commit(&powers, &poly, None, None).unwrap();
    println!("{:?}", commit);
    println!("Done setup");

    return (commit, rand);
}

pub fn prove(inputs: &Vec<Fr>, point: Fr, rand: Randomness<Fr, DensePolynomial<Fr>>) -> (UniPoly_377, Option<UniPoly_377>) {
    let poly = UniPoly_377::from_coefficients_slice(&inputs[..]);
    ark_poly_commit::kzg10::KZG10::<Bls12_377, DensePolynomial<Fr>>::compute_witness_polynomial(&poly, point, &rand).unwrap()
}
