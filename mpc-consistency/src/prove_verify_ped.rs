
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::AffineRepr;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use blake2::Blake2s256;
use crate::ped_pc::BaselinePedPC;

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

fn main() {
    prove_verify::run::<<E as AffineRepr>::ScalarField, UniPoly_377, PCS>();
}

// Implement AddAssignExt trait
impl common::AddAssignExt for <BaselinePedPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377> as ark_poly_commit::PolynomialCommitment<<E as AffineRepr>::ScalarField, UniPoly_377, Sponge_Bls12_377>>::Commitment {
    fn add_assign_ext(&mut self, other: Self) {
        // Implement the addition logic here
        // self.comm = (self.comm + other.comm).into();
        self.comm = self.comm.iter().zip(other.comm.iter()).map(|(a, b)| (*a + b).into()).collect();
    }
}

// Implement AddAssignExtRand
impl<'a> common::AddAssignExtRand<&'a Self> for <BaselinePedPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377> as ark_poly_commit::PolynomialCommitment<<E as AffineRepr>::ScalarField, UniPoly_377, Sponge_Bls12_377>>::Randomness {
    fn add_assign_ext(&mut self, other: &'a Self) {
        // Implement the addition logic here
        self.rand = self.rand.iter().zip(other.rand.iter()).map(|(a, b)| a + b).collect();
    }
}