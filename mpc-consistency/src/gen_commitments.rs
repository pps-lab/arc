

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
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::rand::SeedableRng;
use std::borrow::Cow;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use ark_ec::Group;
// use ark_poly_commit::kzg10::{Commitment, Randomness};
use ark_poly_commit::marlin_pc::UniversalParams;
use clap::{ValueEnum};
use structopt::StructOpt;
use mpc_net::{MpcNet, MpcMultiNet};

use ark_std::{cfg_into_iter, cfg_iter, start_timer, test_rng};
use rayon::prelude::*;

use blake2::digest::generic_array::functional::FunctionalSequence;
use crate::{end_timer, gen_pp};
use log::debug;

use crate::common::{data_dir, FILE_PP, FILE_COM, FILE_DATA, PartyData, get_labeled_poly, get_seeded_rng};
use crate::mpspdz::{parse_shares_from_file, ShareList};
use crate::mpspdz::ShareList::{Sfix, Sint};
use crate::perf_trace_structured::{print_global_stats, print_stats};

// #[derive(PartialEq, Debug, ValueEnum, Clone)]
// pub enum Computation {
//     KzgCommit
// }

#[derive(Debug, StructOpt)]
#[structopt(name = "bench", about = "BenchCommit")]
struct Opt {
    /// Activate debug mode
    // short and long flags (-d, --debug) will be deduced from the field's name
    #[structopt(short, long)]
    debug: bool,

    /// File with list of hosts
    #[structopt(long, parse(from_os_str), default_value = "")]
    hosts: PathBuf,

    /// Which party are you? 0 or 1?
    #[structopt(long, default_value = "0")]
    party: u8,

    /// Computation to perform
    // #[structopt()]
    // computation: Computation,

    #[structopt(long)]
    save: bool,

    #[structopt(long, default_value = "16")]
    precision_f: u32,

    /// Input a
    #[structopt(long)]
    num_args: Option<u64>,

    #[structopt(long, parse(from_os_str), required_unless = "num-args")]
    player_input_binary_path: Option<PathBuf>,
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

pub fn run<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>() {
    debug!("Generating public parameters and exchanging commitments");
    let opt = Opt::from_args();
    if opt.debug {
        env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::init();
    }

    MpcMultiNet::init_from_file(opt.hosts.to_str().unwrap(), opt.party as usize);

    let t_gen_commitments = start_timer!(|| "Generate commitments");

    // let len = opt.num_args;
    // let inputs: Vec<E> = (0..len).map(|_| E::rand(&mut rng)).collect();
    let mut rng = test_rng();
    let mut input_objs: Vec<Vec<E>> = if let Some(filename) = opt.player_input_binary_path {
        // append -format to filename
        let mut filename_format_os = filename.clone().into_os_string();
        filename_format_os.push("-format");
        let filename_format: PathBuf = filename_format_os.into();

        let p = parse_shares_from_file::<PathBuf, f32, i64>(filename, filename_format);
        let inputs_all = p.expect("Failed to parse shares");
        let inputs_int: Vec<Vec<i64>> = cfg_into_iter!(inputs_all).map(|per_obj_list| {
            cfg_into_iter!(per_obj_list).map(|list: ShareList<f32, i64>| -> Vec<i64> {
                match list {
                    Sint(x_list) => {
                        x_list
                    },
                    Sfix(x_list) => {
                        debug!("First 1 value of x_list: {:?}", &x_list[0]);
                        cfg_iter!(x_list).map(|x| {
                            let integer_version = (x * (2u64.pow(opt.precision_f)) as f32).round() as i64;
                            integer_version
                        } ).collect()
                    }
                }
            }).flatten().collect()
        }).collect();

        // Convert to field elements (with negative values)
        cfg_iter!(inputs_int).map(|obj_inputs_int| {
            let iter = cfg_iter!(obj_inputs_int);
            print_type_of(&iter);
            iter.map(|x| {
                // convert to field element
                if *x < 0 {
                    // convert to positive
                    // println!("{} Found negative! {}", opt.party, x);
                    let positive_version = (-1 * x) as u64;
                    // convert to field element
                    -E::from(E::BigInt::from(positive_version))
                } else {
                    // convert to field element
                    E::from(E::BigInt::from(*x as u64))
                }
            }).collect()
        }).collect()

        // debug!("First 10 values of inputs_int: {:?}", &inputs_int[0..10]);

        // let inputs_int: Vec<f32> = parse_shares_from_file(filename, filename_format).expect("Failed to parse shares");
        // Convert f32 to u64 by shifting it by precision_f bits
        // let inputs_int: Vec<u64> = inputs_int.iter().map(|x| (x * (2u64.pow(opt.precision_f)) as f32) as u64).collect();

    } else if let Some(num_args) = opt.num_args {
        vec![(0..num_args).map(|_| E::rand(&mut rng)).collect(); 1]
    } else{
        panic!("Either player_input_binary_path or num_args must be specified");
    };
    let len = input_objs.len();
    println!("Number of input objects: {}", len);
    //
    // let ix = 798928;
    // if ix + 5 < len {
    //     println!("Last 10 inputs: {:?}", &inputs[798928-5..798928+5]);
    // } else {
    // }

    let mut labeled_polys: Vec<LabeledPolynomial<E, P>> = Vec::new();

    // prepend 0 for the first coefficient to keep space for the randomness
    input_objs = input_objs.into_iter().map(|mut obj_inputs| {
        obj_inputs.insert(0, E::zero());
        obj_inputs
    }).collect();

    for mut input in input_objs.iter() {

        let secret_poly = P::from_coefficients_slice(input);

        // 656477353371811962285989500114949133467480537581382083002878563789733456067
        // 1636825947140048331346992203218898965812620282847129052544818971575803068291
        // 3090174033069088738712039487548204773548520248725210634374129034378083167182
        // let point = E::from_str("1");
        // if let Ok(point) = point {
        //     let eval = secret_poly.evaluate(&point);
        //     println!("Party {} Eval poly at point {} {}", opt.party, point.to_string(), eval.to_string());
        // } else {
        //     println!("Failed to parse point");
        // }

        let labeled_poly = get_labeled_poly::<E, P>(secret_poly, None);
        labeled_polys.push(labeled_poly);
    }

    let (ck, vk) = gen_pp::load::<E, P, PCS>();

    debug!("Start");

    let mut compressed_bytes = Vec::new();
    // let (commitment, rand) = kzg::kzg_commit(&pp, &inputs);
    let (labeled_comms, rands) = PCS::commit(&ck, &labeled_polys, Some(&mut rng)).unwrap();

    let comms: Vec<PCS::Commitment> = labeled_comms.into_iter()
        .map(|labeled_comm| labeled_comm.commitment().clone()).collect::<Vec<_>>();
    // Now we communicate the amount of commitments we have
    let num_comms = comms.len() as u8;

    // let all_num_comms_bytes = MpcMultiNet::broadcast_bytes(&num_comms.to_le_bytes());
    // let all_num_comms = all_num_comms_bytes.iter().enumerate().map(|(idx, bytes)| {
    //     u8::from_le_bytes(bytes.as_slice().try_into().expect(format!("Unable to deserialize num_comms {}", idx).as_str()))
    // }).collect::<Vec<_>>();
    //
    // let all_comms_size_bytes = MpcMultiNet::broadcast_bytes(&comms.compressed_size().to_le_bytes());
    // let all_comms_sizes = all_comms_size_bytes.iter().enumerate().map(|(idx, bytes)| {
    //     usize::from_le_bytes(bytes.as_slice().try_into().expect(format!("Unable to deserialize num_comms {}", idx).as_str()))
    // }).collect::<Vec<_>>();
    //
    // let max_num_comms = all_num_comms.into_iter().max().unwrap();
    // let max_comms_size = all_comms_sizes.iter().max().unwrap().clone();
    // println!("Max num comms: {} and size {}", max_num_comms, max_comms_size);
    //

    comms.serialize_compressed(&mut compressed_bytes).unwrap();
    // pad comms to max_comms_size
    // println!("Max comm size is {}", max_comms_size);
    // println!("Current comm size is {}", compressed_bytes.len());
    // let mut padding = vec![0u8; max_comms_size - compressed_bytes.len()];
    // compressed_bytes.append(&mut padding);

    println!("Number of bytes in commitment: {}", compressed_bytes.len());
    // commitment.

    let all_commitments_bytes = MpcMultiNet::broadcast_bytes_unequal(&compressed_bytes);
    let all_commitments: Vec<Vec<PCS::Commitment>> = all_commitments_bytes.iter().enumerate().map(|(idx, bytes)| {
        println!("Current bytes length: {}", bytes.len());
        // let slice = bytes.as_slice()[0..all_comms_sizes[idx]].to_vec();
        let commit_list: Vec<PCS::Commitment> = Vec::<PCS::Commitment>::deserialize_compressed(bytes.as_slice()).expect(format!("Unable to deserialize commitment list for {}", idx).as_str());

        commit_list
    }).collect::<Vec<_>>();

    for (id, masked_poly) in labeled_polys.clone().into_iter().enumerate() {
        let point = E::one();
        let value: E = masked_poly.evaluate(&point);
        println!("Value at point: {}", value);
    }

    // for (idx, comms) in all_commitments.clone().iter().enumerate() {
    //     println!("Party {} has {} commitments", idx, comms.len());
    // }

    // Save all_commitments and rand to file at com_path
    if opt.save {
        let party_data: PartyData<E, P, PoseidonSponge<E>, PCS> = PartyData {
            inputs: input_objs,
            party_id: opt.party,
            rands: rands,
            commitments: all_commitments,
            // max_comms_size: max_comms_size.clone()
        };
        let mut compressed_bytes = Vec::new();
        party_data.serialize_uncompressed(&mut compressed_bytes).unwrap();

        // create path from data_dir and FILE_PP
        let path = data_dir(&format!("{}_{}", FILE_DATA, opt.party));
        println!("Saving at {:?}", path);
        let mut file = File::create(path).expect("Unable to create file");
        file.write_all(&compressed_bytes).expect("Unable to write data");
    }

    end_timer!(t_gen_commitments);

    // println!("Stats: {:#?}", MpcMultiNet::stats());
    print_stats(MpcMultiNet::stats());
    print_global_stats(MpcMultiNet::compute_global_data_sent());
    MpcMultiNet::deinit();
    println!("Done");
}
