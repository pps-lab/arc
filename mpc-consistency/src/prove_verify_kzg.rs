use std::ops::AddAssign;
use ark_bls12_377::Bls12_377;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::marlin_pc::{MarlinKZG10, Randomness};
use ark_poly_commit::PolynomialCommitment;
use num_traits::Zero;
use crate::common::{AddAssignExt, AddAssignExtRand};

mod prove_verify;
mod gen_pp;
mod common;
mod mpspdz;
mod perf_trace_structured;

pub type E = Bls12_377;
pub type F = <E as Pairing>::ScalarField;
pub type UniPoly_377 = DensePolynomial<F>;
pub type Sponge_Bls12_377 = PoseidonSponge<<Bls12_377 as Pairing>::ScalarField>;
pub type PCS = MarlinKZG10<Bls12_377, UniPoly_377, Sponge_Bls12_377>;

fn main() {
    prove_verify::run::<<E as Pairing>::ScalarField, UniPoly_377, PCS>();
}


impl AddAssignExt for <MarlinKZG10<Bls12_377, UniPoly_377, Sponge_Bls12_377> as PolynomialCommitment<<Bls12<ark_bls12_377::Config> as Pairing>::ScalarField, UniPoly_377, Sponge_Bls12_377>>::Commitment {
    fn add_assign_ext(&mut self, other: Self) {
        // Implement the addition logic here
        let mut combined_comm = <Bls12<ark_bls12_377::Config> as Pairing>::G1::zero();
        combined_comm += self.comm.0;
        combined_comm += other.comm.0;
        self.comm.0 = combined_comm.into();
        // self.comm.0 += other.comm.0;
        // if let Some(shifted_comm) = &mut self.shifted_comm {
        //     if let Some(other_shifted_comm) = &other.shifted_comm {
        //         shifted_comm.0 += other_shifted_comm.0;
        //     }
        // }
    }
}
impl<'a> AddAssignExtRand<&'a Self> for <MarlinKZG10<Bls12_377, UniPoly_377, Sponge_Bls12_377> as PolynomialCommitment<<Bls12<ark_bls12_377::Config> as Pairing>::ScalarField, UniPoly_377, Sponge_Bls12_377>>::Randomness {
    fn add_assign_ext(&mut self, other: &'a Self) {
        // Implement the addition logic here
        *self += other;
    }
}
