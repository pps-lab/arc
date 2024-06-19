use ark_bls12_377::Bls12_377;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::marlin_pc::MarlinKZG10;

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
    gen_pp::generate_and_save::<<E as Pairing>::ScalarField, UniPoly_377, PCS>();
}
