

// Commitment storage location

use std::path::PathBuf;
use ark_bls12_377::{ Bls12_377, Fr };
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_crypto_primitives::sponge::poseidon::{PoseidonConfig, PoseidonSponge};
use ark_ec::Group;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::{DenseUVPolynomial, Polynomial};
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::{LabeledPolynomial, PCRandomness, PCUniversalParams, PolynomialCommitment};
// use ark_poly_commit::marlin_pc::{Randomness, UniversalParams};
use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use rand::prelude::StdRng;
use rand::SeedableRng;

pub const data_dir_str: &str = "commitments";

pub fn data_dir(subpath: &str) -> PathBuf {
    let mut path = PathBuf::from(data_dir_str);
    path.push(subpath);
    path
}

pub const FILE_PP: &str = "srs";
pub const FILE_COM: &str = "com_";
pub const FILE_DATA: &str = "data";

pub const PATTERN_SPDZ_RANDOM_POINT: &str = "input_consistency_player_(\\d*)_random_point=(.*)";
pub const PATTERN_SPDZ_OUTPUT_SUM: &str = "input_consistency_player_(\\d*)_output_sum=(.*)";

pub const PATTERN_SPDZ_EVAL: &str = r"input_consistency_player_(\d*)_eval=\((.*),(.*)\)";
pub const PATTERN_SPDZ_RANDOMNESS: &str = r"input_consistency_random_value_(\d*)=(.*)";


#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct PartyData<E: PrimeField, P: DenseUVPolynomial<E>, S: CryptographicSponge, PCS: PolynomialCommitment<E, P, S>> {
    pub inputs: Vec<Vec<E>>,

    pub party_id: u8,

    pub rands: Vec<PCS::Randomness>,
    pub commitments: Vec<Vec<PCS::Commitment>>,
}

pub fn get_seeded_rng() -> StdRng {
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    rand::rngs::StdRng::from_seed(seed)
}

pub fn test_sponge<F: PrimeField>() -> PoseidonSponge<F> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut v = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        v.push(res);
    }
    let config = PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, v, 2, 1);
    PoseidonSponge::new(&config)
}

pub fn get_labeled_poly<E: PrimeField, P: DenseUVPolynomial<E>>(secret_poly: P, label_str: Option<&str>) -> LabeledPolynomial<E, P> {
    let label = String::from(label_str.unwrap_or("secret_poly"));
    return LabeledPolynomial::new(
        label.clone(),
        secret_poly.clone(),
        None,
        Some(1), // we will open a univariate poly at one point
    );
}

// The reason we have to define two traits is because Rust otherwise complains about conflicting implementations.
// error[E0119]: conflicting implementations of trait `AddAssignExt`
pub trait AddAssignExt<Rhs = Self> {
    fn add_assign_ext(&mut self, other: Rhs);
}
pub trait AddAssignExtRand<Rhs = Self> {
    fn add_assign_ext(&mut self, other: Rhs);
}