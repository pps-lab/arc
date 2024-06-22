use ark_std::{cfg_into_iter, collections::{BTreeMap, BTreeSet}, fmt::Debug, hash::Hash, iter::FromIterator, string::{String, ToString}, vec::Vec};
use ark_poly_commit::{BatchLCProof, CHALLENGE_SIZE, DenseUVPolynomial, Error, Evaluations, QuerySet};
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, LinearCombination};
use ark_poly_commit::{PCCommitterKey, PCRandomness, PCUniversalParams, PolynomialCommitment};

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use ark_std::{convert::TryInto, end_timer, format, marker::PhantomData, ops::Mul, start_timer, vec};

use rayon::prelude::*;

mod data_structures;
pub use data_structures::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use ark_poly_commit::challenge::ChallengeGenerator;
use ark_crypto_primitives::sponge::CryptographicSponge;
use blake2::digest::Digest;

/// This is an implementation of a 'baseline' commitment scheme based on individual Pedersen commitments.
/// This is technically not a polynmial commitment scheme because it is not succinct.
pub struct BaselinePedPC<
    G: AffineRepr,
    D: Digest,
    P: DenseUVPolynomial<G::ScalarField>,
    S: CryptographicSponge,
> {
    _projective: PhantomData<G>,
    _digest: PhantomData<D>,
    _poly: PhantomData<P>,
    _sponge: PhantomData<S>,
}

impl<G, D, P, S> BaselinePedPC<G, D, P, S>
where
    G: AffineRepr,
    D: Digest,
    P: DenseUVPolynomial<G::ScalarField>,
    S: CryptographicSponge,
{
    /// `PROTOCOL_NAME` is used as a seed for the setup function.
    pub const PROTOCOL_NAME: &'static [u8] = b"PC-DL-2020-PED";

    fn print_type_of<T>(_: &T) {
        println!("{}", std::any::type_name::<T>())
    }

    /// Create a Pedersen commitment to `scalars` using the commitment key `comm_key`.
    /// Optionally, randomize the commitment using `hiding_generator` and `randomizer`.
    fn cm_commit(
        comm_key: G,
        scalars: &[G::ScalarField],
        hiding_generator: G,
        randomizers: &[G::ScalarField],
    ) -> Vec<G::Group> {
        // let scalars_bigint = ark_std::cfg_iter!(scalars)
        //     .map(|s| s.into_bigint())
        //     .collect::<Vec<_>>();

        let timerCommit = start_timer!(|| format!("Computing commitment to {} scalars and {} random values!!", scalars.len(), randomizers.len()));

        let iter = cfg_into_iter!(scalars);
        Self::print_type_of(&iter);
        let comm = iter.zip(randomizers).map(|(scalar, rand)| {
            comm_key.into_group().mul(scalar) + hiding_generator.into_group().mul(rand)
        }).collect();

        // let mut comm = <G::Group as VariableBaseMSM>::msm_bigint(comm_key, &scalars_bigint);

        end_timer!(timerCommit);

        comm
    }

    fn compute_random_oracle_challenge(bytes: &[u8]) -> G::ScalarField {
        let mut i = 0u64;
        let mut challenge = None;
        while challenge.is_none() {
            let mut hash_input = bytes.to_vec();
            hash_input.extend(i.to_le_bytes());
            let hash = D::digest(&hash_input.as_slice());
            challenge = <G::ScalarField as Field>::from_random_bytes(&hash);

            i += 1;
        }

        challenge.unwrap()
    }

    fn sample_generators(num_generators: usize) -> Vec<G> {
        let generators: Vec<_> = cfg_into_iter!(0..num_generators)
            .map(|i| {
                let i = i as u64;
                let mut hash =
                    D::digest([Self::PROTOCOL_NAME, &i.to_le_bytes()].concat().as_slice());
                let mut g = G::from_random_bytes(&hash);
                let mut j = 0u64;
                while g.is_none() {
                    // PROTOCOL NAME, i, j
                    let mut bytes = Self::PROTOCOL_NAME.to_vec();
                    bytes.extend(i.to_le_bytes());
                    bytes.extend(j.to_le_bytes());
                    hash = D::digest(bytes.as_slice());
                    g = G::from_random_bytes(&hash);
                    j += 1;
                }
                let generator = g.unwrap();
                generator.mul_by_cofactor_to_group()
            })
            .collect();

        G::Group::normalize_batch(&generators)
    }
}

impl<G, D, P, S> PolynomialCommitment<G::ScalarField, P, S> for BaselinePedPC<G, D, P, S>
where
    G: AffineRepr,
    G::Group: VariableBaseMSM<MulBase = G>,
    D: Digest,
    P: DenseUVPolynomial<G::ScalarField, Point = G::ScalarField>,
    S: CryptographicSponge,
{
    type UniversalParams = UniversalParams<G>;
    type CommitterKey = CommitterKey<G>;
    type VerifierKey = VerifierKey<G>;
    type PreparedVerifierKey = PreparedVerifierKey<G>;
    type Commitment = Commitment<G>;
    type PreparedCommitment = PreparedCommitment<G>;
    type Randomness = Randomness<G>;
    type Proof = Proof<G>;
    type BatchProof = Vec<Self::Proof>;
    type Error = Error;

    fn setup<R: RngCore>(
        max_degree: usize,
        _: Option<usize>,
        _rng: &mut R,
    ) -> Result<Self::UniversalParams, Self::Error> {
        // Ensure that max_degree + 1 is a power of 2
        let max_degree = (max_degree + 1).next_power_of_two() - 1;

        let setup_time = start_timer!(|| format!("Sampling {} generators", 2));
        let mut generators = Self::sample_generators(2);
        end_timer!(setup_time);

        let pp = UniversalParams {
            comm_key: generators[0],
            hiding_comm_key: generators[1]
        };

        Ok(pp)
    }

    fn trim(
        pp: &Self::UniversalParams,
        supported_degree: usize,
        _supported_hiding_bound: usize,
        _enforced_degree_bounds: Option<&[usize]>,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error> {
        // Ensure that supported_degree + 1 is a power of two
        // let supported_degree = (supported_degree + 1).next_power_of_two() - 1;
        // if supported_degree > pp.max_degree() {
        //     return Err(Error::TrimmingDegreeTooLarge);
        // }

        let trim_time =
            start_timer!(|| format!("Trimming to supported degree of {}", supported_degree));

        let ck = CommitterKey {
            comm_key: pp.comm_key,
            hiding_comm_key: pp.hiding_comm_key
        };

        let vk = VerifierKey {
            comm_key: pp.comm_key,
            hiding_comm_key: pp.hiding_comm_key
        };

        end_timer!(trim_time);

        Ok((ck, vk))
    }

    /// Outputs a commitment to `polynomial`.
    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField, P>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<Self::Randomness>,
        ),
        Self::Error,
    >
    where
        P: 'a,
    {
        let rng = &mut ark_poly_commit::optional_rng::OptionalRng(rng);
        let mut comms = Vec::new();
        let mut rands = Vec::new();

        let commit_time = start_timer!(|| "Committing to polynomials");
        for labeled_polynomial in polynomials {

            let polynomial: &P = labeled_polynomial.polynomial();
            let label = labeled_polynomial.label();
            let hiding_bound = labeled_polynomial.degree();
            let degree_bound = labeled_polynomial.degree();

            let commit_time = start_timer!(|| format!(
                "Polynomial {} of degree {}, degree bound {:?}, and hiding bound {:?}",
                label,
                polynomial.degree(),
                degree_bound,
                hiding_bound,
            ));

            let randomness = Randomness::rand(hiding_bound + 1, false, None, rng);

            let comm: Vec<G> = Self::cm_commit(
                ck.comm_key,
                &polynomial.coeffs(),
                ck.hiding_comm_key,
                &randomness.rand,
            )
                .into_iter().map(|x| x.into()).collect();

            let commitment = Commitment { comm };
            let labeled_comm = LabeledCommitment::new(label.to_string(), commitment, None);

            comms.push(labeled_comm);
            rands.push(randomness);

            end_timer!(commit_time);
        }

        end_timer!(commit_time);
        Ok((comms, rands))
    }

    fn open<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rands: impl IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
    where
        Self::Commitment: 'a,
        Self::Randomness: 'a,
        P: 'a,
    {
        // let mut combined_polynomial = P::zero();
        // let mut combined_rand = G::ScalarField::zero();
        // let mut combined_commitment_proj = G::Group::zero();
        //
        // let mut has_hiding = false;

            let mut polys_iter = labeled_polynomials.into_iter();
            let mut rands_iter = rands.into_iter();
            // let comms_iter = commitments.into_iter();

            // we want to:
            // 1. compute com(rho) where rho is the evaluation of our polynomial at point
            // 2. compute evaluation of the commitment at point rho
            // 3. proof that the resulting commitment is a commitment to 0
        {
            let combined_polynomial = polys_iter.next().unwrap();
            let combined_rand = rands_iter.next().unwrap();

            // Powers of z
            let d = combined_polynomial.degree();
            let mut z: Vec<G::ScalarField> = Vec::with_capacity(d + 1);
            let mut cur_z: G::ScalarField = G::ScalarField::one();
            for _ in 0..(d + 1) {
                z.push(cur_z);
                cur_z *= point;
            }
            let mut z = z.as_mut_slice();

            // Compute combination of randomnesses
            let mut rho_opening = G::ScalarField::zero();
            for i in 0..(d + 1) {
                rho_opening += z[i] * &combined_rand.rand[i];
            }

            Ok(Proof {
                combined_opening: rho_opening
            })
        }
    }

    fn check<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Self::Proof,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        let check_time = start_timer!(|| "Checking evaluations");

        let mut commitments_iter = commitments.into_iter();
        let mut values_iter = values.into_iter();


        // assert!(commitments_iter.len() == 1, "Only one polynomial opening is supported at a time!");

        // we want to:
        // 1. compute com(rho) where rho is the evaluation of our polynomial at point
        // 2. compute evaluation of the commitment at point rho
        // 3. proof that the resulting commitment is a commitment to 0
        {
            let combined_commitments = commitments_iter.next().unwrap();
            // println!("commitments {:?}", combined_commitments.commitment().comm);
            let combined_values = values_iter.next().unwrap();

            // Powers of z
            let d = combined_commitments.commitment().comm.len();
            println!("commitment size {:?}", d);
            let mut z: Vec<G::ScalarField> = Vec::with_capacity(d + 1);
            let mut cur_z: G::ScalarField = G::ScalarField::one();
            for _ in 0..d {
                z.push(cur_z);
                cur_z *= point;
            }
            let mut z = z.as_mut_slice();

            // let mut rho_com = G::Group::zero();
            // for i in 0..d {
            //     rho_com += &combined_commitments.commitment().comm[i].mul(z[i]);
            // }

            // Do MSM here:
            let scalars_bigint = ark_std::cfg_iter!(z)
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();

            let rho_com = <G::Group as VariableBaseMSM>::msm_bigint(&combined_commitments.commitment().comm, &scalars_bigint);

            // Check that rho_com is to zero
            let rho_com_check = vk.comm_key.mul(combined_values) +
                vk.hiding_comm_key.mul(proof.combined_opening);
            let equal = rho_com == rho_com_check;
            end_timer!(check_time);

            Ok(equal)
        }
    }

    fn batch_check<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        values: &Evaluations<G::ScalarField, P::Point>,
        proof: &Self::BatchProof,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        panic!("Not implemented!");
        // let commitments: BTreeMap<_, _> = commitments.into_iter().map(|c| (c.label(), c)).collect();
        // let mut query_to_labels_map = BTreeMap::new();
        //
        // for (label, (point_label, point)) in query_set.iter() {
        //     let labels = query_to_labels_map
        //         .entry(point_label)
        //         .or_insert((point, BTreeSet::new()));
        //     labels.1.insert(label);
        // }
        //
        // assert_eq!(proof.len(), query_to_labels_map.len());
        //
        // let mut randomizer = G::ScalarField::one();
        //
        // let mut combined_check_poly = P::zero();
        // let mut combined_final_key = G::Group::zero();
        //
        // for ((_point_label, (point, labels)), p) in query_to_labels_map.into_iter().zip(proof) {
        //     let lc_time =
        //         start_timer!(|| format!("Randomly combining {} commitments", labels.len()));
        //     let mut comms: Vec<&'_ LabeledCommitment<_>> = Vec::new();
        //     let mut vals = Vec::new();
        //     for label in labels.into_iter() {
        //         let commitment = commitments.get(label).ok_or(Error::MissingPolynomial {
        //             label: label.to_string(),
        //         })?;
        //
        //         let v_i = values
        //             .get(&(label.clone(), *point))
        //             .ok_or(Error::MissingEvaluation {
        //                 label: label.to_string(),
        //             })?;
        //
        //         comms.push(commitment);
        //         vals.push(*v_i);
        //     }
        //
        //     let check_poly = Self::succinct_check(
        //         vk,
        //         comms.into_iter(),
        //         *point,
        //         vals.into_iter(),
        //         p,
        //         opening_challenges,
        //     );
        //
        //     if check_poly.is_none() {
        //         return Ok(false);
        //     }
        //
        //     let check_poly = P::from_coefficients_vec(check_poly.unwrap().compute_coeffs());
        //     combined_check_poly += (randomizer, &check_poly);
        //     combined_final_key += &p.final_comm_key.mul(randomizer);
        //
        //     randomizer = u128::rand(rng).into();
        //     end_timer!(lc_time);
        // }
        //
        // let proof_time = start_timer!(|| "Checking batched proof");
        // let final_key = Self::cm_commit(
        //     vk.comm_key.as_slice(),
        //     combined_check_poly.coeffs(),
        //     None,
        //     None,
        // );
        // if !(final_key - &combined_final_key).is_zero() {
        //     return Ok(false);
        // }
        //
        // end_timer!(proof_time);
        //
        // Ok(true)
    }

    fn open_combinations<'a>(
        ck: &Self::CommitterKey,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<G::ScalarField>>,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rands: impl IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<BatchLCProof<G::ScalarField, Self::BatchProof>, Self::Error>
    where
        Self::Randomness: 'a,
        Self::Commitment: 'a,
        P: 'a,
    {
        panic!("Not implemented!");

        // let label_poly_map = polynomials
        //     .into_iter()
        //     .zip(rands)
        //     .zip(commitments)
        //     .map(|((p, r), c)| (p.label(), (p, r, c)))
        //     .collect::<BTreeMap<_, _>>();
        //
        // let mut lc_polynomials = Vec::new();
        // let mut lc_randomness = Vec::new();
        // let mut lc_commitments = Vec::new();
        // let mut lc_info = Vec::new();
        //
        // for lc in linear_combinations {
        //     let lc_label = lc.label().clone();
        //     let mut poly = P::zero();
        //     let mut degree_bound = None;
        //     let mut hiding_bound = None;
        //
        //     let mut combined_comm = G::Group::zero();
        //     let mut combined_shifted_comm: Option<G::Group> = None;
        //
        //     let mut combined_rand = G::ScalarField::zero();
        //     let mut combined_shifted_rand: Option<G::ScalarField> = None;
        //
        //     let num_polys = lc.len();
        //     for (coeff, label) in lc.iter().filter(|(_, l)| !l.is_one()) {
        //         let label: &String = label.try_into().expect("cannot be one!");
        //         let &(cur_poly, cur_rand, cur_comm) =
        //             label_poly_map.get(label).ok_or(Error::MissingPolynomial {
        //                 label: label.to_string(),
        //             })?;
        //
        //         if num_polys == 1 && cur_poly.degree_bound().is_some() {
        //             assert!(
        //                 coeff.is_one(),
        //                 "Coefficient must be one for degree-bounded equations"
        //             );
        //             degree_bound = cur_poly.degree_bound();
        //         } else if cur_poly.degree_bound().is_some() {
        //             eprintln!("Degree bound when number of equations is non-zero");
        //             return Err(Self::Error::EquationHasDegreeBounds(lc_label));
        //         }
        //
        //         // Some(_) > None, always.
        //         hiding_bound = core::cmp::max(hiding_bound, cur_poly.hiding_bound());
        //         poly += (*coeff, cur_poly.polynomial());
        //
        //         combined_rand += &(cur_rand.rand * coeff);
        //         combined_shifted_rand = Self::combine_shifted_rand(
        //             combined_shifted_rand,
        //             cur_rand.shifted_rand,
        //             *coeff,
        //         );
        //
        //         let commitment = cur_comm.commitment();
        //         combined_comm += &commitment.comm.mul(*coeff);
        //         combined_shifted_comm = Self::combine_shifted_comm(
        //             combined_shifted_comm,
        //             commitment.shifted_comm,
        //             *coeff,
        //         );
        //     }
        //
        //     let lc_poly =
        //         LabeledPolynomial::new(lc_label.clone(), poly, degree_bound, hiding_bound);
        //     lc_polynomials.push(lc_poly);
        //     lc_randomness.push(Randomness {
        //         rand: combined_rand,
        //         shifted_rand: combined_shifted_rand,
        //     });
        //
        //     lc_commitments.push(combined_comm);
        //     if let Some(combined_shifted_comm) = combined_shifted_comm {
        //         lc_commitments.push(combined_shifted_comm);
        //     }
        //
        //     lc_info.push((lc_label, degree_bound));
        // }
        //
        // let lc_commitments = Self::construct_labeled_commitments(&lc_info, &lc_commitments);
        //
        // let proof = Self::batch_open(
        //     ck,
        //     lc_polynomials.iter(),
        //     lc_commitments.iter(),
        //     &query_set,
        //     opening_challenges,
        //     lc_randomness.iter(),
        //     rng,
        // )?;
        // Ok(BatchLCProof { proof, evals: None })
    }

    /// Checks that `values` are the true evaluations at `query_set` of the polynomials
    /// committed in `labeled_commitments`.
    fn check_combinations<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        eqn_query_set: &QuerySet<P::Point>,
        eqn_evaluations: &Evaluations<P::Point, G::ScalarField>,
        proof: &BatchLCProof<G::ScalarField, Self::BatchProof>,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        panic!("Not implemented!");

        // let BatchLCProof { proof, .. } = proof;
        // let label_comm_map = commitments
        //     .into_iter()
        //     .map(|c| (c.label(), c))
        //     .collect::<BTreeMap<_, _>>();
        //
        // let n_com_size = commitments[0].comm.len();
        // // assert all commitments have the same size
        // assert!(commitments.iter().all(|c| c.comm.len() == n_com_size));
        //
        // let mut lc_commitments = Vec::new();
        // let mut lc_info = Vec::new();
        // let mut evaluations = eqn_evaluations.clone();
        // for lc in linear_combinations {
        //     let lc_label = lc.label().clone();
        //     let num_polys = lc.len();
        //
        //     let mut degree_bound = None;
        //     let mut combined_comm = vec![G::Group::zero(), n_com_size];
        //     let mut combined_shifted_comm: Option<G::Group> = None;
        //
        //     for (coeff, label) in lc.iter() {
        //         if label.is_one() {
        //             for (&(ref label, _), ref mut eval) in evaluations.iter_mut() {
        //                 if label == &lc_label {
        //                     **eval -= coeff;
        //                 }
        //             }
        //         } else {
        //             let label: &String = label.try_into().unwrap();
        //             let &cur_comm = label_comm_map.get(label).ok_or(Error::MissingPolynomial {
        //                 label: label.to_string(),
        //             })?;
        //
        //             if num_polys == 1 && cur_comm.degree_bound().is_some() {
        //                 assert!(
        //                     coeff.is_one(),
        //                     "Coefficient must be one for degree-bounded equations"
        //                 );
        //                 degree_bound = cur_comm.degree_bound();
        //             } else if cur_comm.degree_bound().is_some() {
        //                 return Err(Self::Error::EquationHasDegreeBounds(lc_label));
        //             }
        //
        //             let commitment = cur_comm.commitment();
        //             combined_comm += &commitment.comm.mul(*coeff);
        //             combined_shifted_comm = Self::combine_shifted_comm(
        //                 combined_shifted_comm,
        //                 commitment.shifted_comm,
        //                 *coeff,
        //             );
        //         }
        //     }
        //
        //     lc_commitments.push(combined_comm);
        //
        //     if let Some(combined_shifted_comm) = combined_shifted_comm {
        //         lc_commitments.push(combined_shifted_comm);
        //     }
        //
        //     lc_info.push((lc_label, degree_bound));
        // }
        //
        // let lc_commitments = Self::construct_labeled_commitments(&lc_info, &lc_commitments);
        //
        // Self::batch_check(
        //     vk,
        //     &lc_commitments,
        //     &eqn_query_set,
        //     &evaluations,
        //     proof,
        //     opening_challenges,
        //     rng,
        // )
    }
}
