
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::AffineRepr;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use blake2::Blake2s256;
use log::debug;
use crate::ped_pc::BaselinePedPC;

mod gen_commitments;
mod gen_pp;
mod common;
mod mpspdz;
mod perf_trace_structured;

mod ped_pc;

pub type E = <Bls12<ark_bls12_377::Config> as Pairing>::G1Affine;
pub type F = <E as AffineRepr>::ScalarField;
pub type UniPoly_377 = DensePolynomial<F>;
pub type Sponge_Bls12_377 = PoseidonSponge<<E as AffineRepr>::ScalarField>;

pub type PCS = BaselinePedPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377>;


fn main() {
    debug!("Warning: This code only supports inputs of equal size for each party.\
        If the script hangs after committing to the polynomials,\
        it is because the commitments are likely of different size \
        (broadcasts expects them to be the same size).");

    gen_commitments::run::<<E as AffineRepr>::ScalarField, UniPoly_377, PCS>();
}
