

use ark_poly_commit::{Polynomial, LabeledPolynomial, PolynomialCommitment, QuerySet, Evaluations, challenge::ChallengeGenerator};
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
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::SeedableRng;
use std::borrow::Cow;
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use ark_ec::Group;
// use ark_poly_commit::kzg10::{Commitment, Randomness};
use ark_poly_commit::marlin_pc::UniversalParams;
use clap::{ValueEnum};
use structopt::StructOpt;

use ark_std::{start_timer, test_rng};
use crate::end_timer;
use log::debug;

use crate::common::{data_dir, FILE_PP, FILE_COM, FILE_DATA, PartyData, get_labeled_poly, get_seeded_rng};
use crate::mpspdz::{parse_shares_from_file};
use crate::perf_trace_structured::print_stats;

// #[derive(PartialEq, Debug, ValueEnum, Clone)]
// pub enum Computation {
//     KzgCommit
// }

#[derive(Debug, StructOpt)]
#[structopt(name = "bench", about = "BenchCommit")]
struct Opt {
    /// Input a
    #[structopt(long)]
    num_args: u64,
}


pub fn generate_and_save<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>() {
    let opt = Opt::from_args();
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    let t_gen_pp = start_timer!(|| "Generating pp");

    let len = opt.num_args;
    println!("Generating public parameters for {} args", opt.num_args);

    let mut rng = test_rng();

    let t_pp = start_timer!(|| "Setup pp");
    let pp = PCS::setup(len as usize, None, &mut rng).unwrap();
    end_timer!(t_pp);

    let t_trim = start_timer!(|| "Trim pp");
    let (ck, vk) = PCS::trim(&pp, len as usize, 1, None).unwrap();
    end_timer!(t_trim);

    let mut compressed_bytes = Vec::new();
    (ck, vk).serialize_uncompressed(&mut compressed_bytes).unwrap();

    let path = data_dir(&format!("{}", FILE_PP));
    println!("Saving at {:?}", path);
    let mut file = File::create(path).expect("Unable to create file");
    file.write_all(&compressed_bytes).expect("Unable to write data");

    end_timer!(t_gen_pp);

    println!("Done");
}

pub fn load<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>() -> (PCS::CommitterKey, PCS::VerifierKey) {

    let data_file = data_dir(&format!("{}", FILE_PP));
    debug!("Loading pp from file: {}", data_file.to_str().unwrap());
    if !data_file.exists() {
        panic!("PP file does not exist. Run gen_pp first.");
    }
    let mut file = File::open(data_file).expect("Unable to open file");
    let mut compressed_bytes = Vec::new();
    file.read_to_end(&mut compressed_bytes).expect("Unable to read data");

    return <(PCS::CommitterKey, PCS::VerifierKey)>::deserialize_uncompressed_unchecked(compressed_bytes.as_slice()).expect("Unable to deserialize ck and vk data");

}
