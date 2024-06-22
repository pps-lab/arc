use ark_poly_commit::*;
use ark_poly_commit::{PCCommitterKey, PCVerifierKey};
use ark_ec::AffineRepr;
use ark_ff::{Field, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use ark_std::vec;
use derivative::Derivative;

/// `UniversalParams` are the universal parameters for the inner product arg scheme.
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
pub struct UniversalParams<G: AffineRepr> {
    /// The key used to commit to polynomials.
    pub comm_key: G,
    pub hiding_comm_key: G,
}

impl<G: AffineRepr> PCUniversalParams for UniversalParams<G> {
    fn max_degree(&self) -> usize {
        1
    }
}

/// `CommitterKey` is used to commit to, and create evaluation proofs for, a given
/// polynomial.
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = "")
)]
pub struct CommitterKey<G: AffineRepr> {
    /// The key used to commit to polynomials.
    pub comm_key: G,
    pub hiding_comm_key: G

}

impl<G: AffineRepr> PCCommitterKey for CommitterKey<G> {
    fn max_degree(&self) -> usize {
        1
    }
    fn supported_degree(&self) -> usize {
        1
    }
}

/// `VerifierKey` is used to check evaluation proofs for a given commitment.
pub type VerifierKey<G> = CommitterKey<G>;

impl<G: AffineRepr> PCVerifierKey for VerifierKey<G> {
    fn max_degree(&self) -> usize {
        1
    }

    fn supported_degree(&self) -> usize {
        1
    }
}

/// Nothing to do to prepare this verifier key (for now).
pub type PreparedVerifierKey<G> = VerifierKey<G>;

impl<G: AffineRepr> PCPreparedVerifierKey<VerifierKey<G>> for PreparedVerifierKey<G> {
    /// prepare `PreparedVerifierKey` from `VerifierKey`
    fn prepare(vk: &VerifierKey<G>) -> Self {
        vk.clone()
    }
}

/// Commitment to a polynomial that optionally enforces a degree bound.
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct Commitment<G: AffineRepr> {
    /// A Pedersen commitment to the polynomial.
    pub comm: Vec<G>,
}

impl<G: AffineRepr> PCCommitment for Commitment<G> {
    #[inline]
    fn empty() -> Self {
        Commitment {
            comm: Vec::new()
        }
    }

    fn has_degree_bound(&self) -> bool {
        false
    }
}


/// Nothing to do to prepare this commitment (for now).
pub type PreparedCommitment<E> = Commitment<E>;

impl<G: AffineRepr> PCPreparedCommitment<Commitment<G>> for PreparedCommitment<G> {
    /// prepare `PreparedCommitment` from `Commitment`
    fn prepare(vk: &Commitment<G>) -> Self {
        vk.clone()
    }
}


/// `Randomness` hides the polynomial inside a commitment and is outputted by `InnerProductArg::commit`.
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct Randomness<G: AffineRepr> {
    /// Randomness is some scalar field element.
    pub rand: Vec<G::ScalarField>,
}

impl<G: AffineRepr> PCRandomness for Randomness<G> {
    fn empty() -> Self {
        Self {
            rand: Vec::new(),
        }
    }

    fn rand<R: RngCore>(h: usize, has_degree_bound: bool, _: Option<usize>, rng: &mut R) -> Self {
        let rand = (0..h).map(|_| G::ScalarField::rand(rng)).collect();

        Self { rand }
    }
}

/// `Proof` is an evaluation proof that is output by `InnerProductArg::open`.
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = "")
)]
pub struct Proof<G: AffineRepr> {
    /// Vector of left elements for each of the log_d iterations in `open`
    pub combined_opening: G::ScalarField
}