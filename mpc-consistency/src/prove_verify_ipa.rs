
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::AffineRepr;
use ark_ec::pairing::Pairing;
// use ark_ed_on_bls12_377::{EdwardsAffine, Fr, EdwardsConfig};
use ark_bls12_377::Bls12_377;
use ark_ec::bls12::Bls12;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_secp256k1::{Affine, Fr};
use blake2::Blake2s256;

mod prove_verify;
mod gen_pp;
mod common;
mod mpspdz;
mod perf_trace_structured;

// pub type E = <Bls12<ark_bls12_377::Config> as Pairing>::G1Affine;
// pub type F = <E as AffineRepr>::ScalarField;
// pub type UniPoly_377 = DensePolynomial<F>;
// pub type Sponge_Bls12_377 = PoseidonSponge<<E as AffineRepr>::ScalarField>;

pub type E = Affine;
pub type UniPoly_377 = DensePolynomial<Fr>;
pub type Sponge_Bls12_377 = PoseidonSponge<<E as AffineRepr>::ScalarField>;

// pub type E = G1Affine;
// pub type UniPoly_377 = DensePolynomial<Fr>;
// pub type Sponge_Bls12_377 = PoseidonSponge<<E as AffineRepr>::ScalarField>;

// We want to get the affine repr of bls12 377 scalar field, but not bn

pub type PCS = InnerProductArgPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377>;

fn main() {
    prove_verify::run::<<E as AffineRepr>::ScalarField, UniPoly_377, PCS>();
}


// Implement AddAssignExt
impl common::AddAssignExt for <InnerProductArgPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377> as ark_poly_commit::PolynomialCommitment<<E as AffineRepr>::ScalarField, UniPoly_377, Sponge_Bls12_377>>::Commitment {
    fn add_assign_ext(&mut self, other: Self) {
        // Implement the addition logic here
        self.comm = (self.comm + other.comm).into();
        if let Some(shifted_comm) = &mut self.shifted_comm {
            if let Some(other_shifted_comm) = &other.shifted_comm {
                self.shifted_comm = Some((*shifted_comm + other_shifted_comm).into());
            }
        }
    }
}

// Implement AddAssignExtRand
impl<'a> common::AddAssignExtRand<&'a Self> for <InnerProductArgPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377> as ark_poly_commit::PolynomialCommitment<<E as AffineRepr>::ScalarField, UniPoly_377, Sponge_Bls12_377>>::Randomness {
    fn add_assign_ext(&mut self, other: &'a Self) {
        // Implement the addition logic here
        self.rand += other.rand;
        if let Some(shifted_rand) = &mut self.shifted_rand {
            if let Some(other_shifted_rand) = &other.shifted_rand {
                self.shifted_rand = Some(*shifted_rand + other_shifted_rand);
            }
        }
    }
}