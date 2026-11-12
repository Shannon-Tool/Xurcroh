use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use thiserror::Error;

pub const RING_MODULUS: i128 = 1i128 << 32;
pub const PRIME_MODULUS: i128 = RING_MODULUS - 5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i64>,
}

#[derive(Debug, Error)]
pub enum ArithmeticError {
    #[error("matrix dimensions do not match")]
    Dimension,
    #[error("checked arithmetic overflow")]
    Overflow,
    #[error("invalid share count")]
    ShareCount,
}

#[derive(Debug, Error)]
#[error("role ({activation_index}, {weight_index}) at {provider_id} failed: {reason}")]
pub struct RoleExecutionError {
    pub activation_index: usize,
    pub weight_index: usize,
    pub provider_id: String,
    pub reason: String,
}

#[derive(Clone)]
pub struct WeightShare {
    pub weight_index: usize,
    pub matrix: Matrix,
    pub challenge: FreivaldsChallenge,
    pub provider_id: String,
    pub replica_provider_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FreivaldsChallenge {
    pub r: Vec<i64>,
    pub v: Vec<i64>,
    pub prime: i128,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<i64>) -> Result<Self, ArithmeticError> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(ArithmeticError::Dimension);
        }
        Ok(Self { rows, cols, data })
    }

    fn value(&self, row: usize, col: usize) -> i64 {
        self.data[row * self.cols + col]
    }

    fn from_values(rows: usize, cols: usize, values: impl Iterator<Item = i64>) -> Self {
        Self {
            rows,
            cols,
            data: values.collect(),
        }
    }

    pub fn matvec(&self, vector: &[i64], modulus: i128) -> Result<Vec<i64>, ArithmeticError> {
        if vector.len() != self.rows {
            return Err(ArithmeticError::Dimension);
        }
        (0..self.cols)
            .map(|col| {
                let mut sum = 0i128;
                for (row, value) in vector.iter().enumerate() {
                    let product = i128::from(*value)
                        .checked_mul(i128::from(self.value(row, col)))
                        .ok_or(ArithmeticError::Overflow)?;
                    sum = sum.checked_add(product).ok_or(ArithmeticError::Overflow)?;
                }
                Ok((sum.rem_euclid(modulus)) as i64)
            })
            .collect()
    }

    pub fn transpose_matvec(
        &self,
        vector: &[i64],
        modulus: i128,
    ) -> Result<Vec<i64>, ArithmeticError> {
        self.transpose_matvec_exact(vector).map(|values| {
            values
                .into_iter()
                .map(|value| value.rem_euclid(modulus) as i64)
                .collect()
        })
    }

    pub fn transpose_matvec_exact(&self, vector: &[i64]) -> Result<Vec<i128>, ArithmeticError> {
        if vector.len() != self.cols {
            return Err(ArithmeticError::Dimension);
        }
        (0..self.rows)
            .map(|row| {
                let mut sum = 0i128;
                for (col, value) in vector.iter().enumerate() {
                    let product = i128::from(*value)
                        .checked_mul(i128::from(self.value(row, col)))
                        .ok_or(ArithmeticError::Overflow)?;
                    sum = sum.checked_add(product).ok_or(ArithmeticError::Overflow)?;
                }
                Ok(sum)
            })
            .collect()
    }

    fn random(rows: usize, cols: usize, seeded: Option<[u8; 32]>) -> Self {
        let mut rng = match seeded {
            Some(seed) => ChaCha20Rng::from_seed(seed),
            None => ChaCha20Rng::from_entropy(),
        };
        Self::from_values(
            rows,
            cols,
            (0..rows * cols).map(|_| (rng.next_u64() % RING_MODULUS as u64) as i64),
        )
    }

    fn add_mod(&self, other: &Self, modulus: i128) -> Self {
        Self::from_values(
            self.rows,
            self.cols,
            self.data.iter().zip(&other.data).map(|(left, right)| {
                (i128::from(*left) + i128::from(*right)).rem_euclid(modulus) as i64
            }),
        )
    }

    fn sub_mod(&self, other: &Self, modulus: i128) -> Self {
        Self::from_values(
            self.rows,
            self.cols,
            self.data.iter().zip(&other.data).map(|(left, right)| {
                (i128::from(*left) - i128::from(*right)).rem_euclid(modulus) as i64
            }),
        )
    }
}

impl std::ops::Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols));
        let combined = self.add_mod(&rhs, RING_MODULUS);
        Self::from_values(
            combined.rows,
            combined.cols,
            combined.data.into_iter().map(|value| {
                if value >= RING_MODULUS as i64 / 2 {
                    value - RING_MODULUS as i64
                } else {
                    value
                }
            }),
        )
    }
}

pub fn split_matrix(secret: &Matrix, count: usize) -> Vec<Matrix> {
    assert!(count >= 2, "{}", ArithmeticError::ShareCount);
    let mut shares = (0..count - 1)
        .map(|_| Matrix::random(secret.rows, secret.cols, None))
        .collect::<Vec<_>>();
    let total = shares.iter().fold(
        Matrix::new(secret.rows, secret.cols, vec![0; secret.data.len()]).unwrap(),
        |sum, share| sum.add_mod(share, RING_MODULUS),
    );
    shares.push(secret.sub_mod(&total, RING_MODULUS));
    shares
}

pub fn seeded_split_matrix(secret: &Matrix, count: usize, seed: [u8; 32]) -> Vec<Matrix> {
    assert!(count >= 2, "{}", ArithmeticError::ShareCount);
    let mut shares = (0..count - 1)
        .map(|_| Matrix::random(secret.rows, secret.cols, Some(seed)))
        .collect::<Vec<_>>();
    let total = shares.iter().fold(
        Matrix::new(secret.rows, secret.cols, vec![0; secret.data.len()]).unwrap(),
        |sum, share| sum.add_mod(share, RING_MODULUS),
    );
    shares.push(secret.sub_mod(&total, RING_MODULUS));
    shares
}

pub fn split_vector(secret: &[i64], count: usize) -> Vec<Vec<i64>> {
    assert!(count >= 2, "{}", ArithmeticError::ShareCount);
    let mut rng = ChaCha20Rng::from_entropy();
    let mut shares = (0..count - 1)
        .map(|_| {
            secret
                .iter()
                .map(|_| (rng.next_u64() % RING_MODULUS as u64) as i64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let total = (0..secret.len())
        .map(|index| {
            shares
                .iter()
                .map(|share| i128::from(share[index]))
                .sum::<i128>()
                .rem_euclid(RING_MODULUS) as i64
        })
        .collect::<Vec<_>>();
    shares.push(
        secret
            .iter()
            .zip(total)
            .map(|(value, mask)| {
                (i128::from(*value) - i128::from(mask)).rem_euclid(RING_MODULUS) as i64
            })
            .collect(),
    );
    shares
}

pub fn refresh_shares(first: &Matrix, second: &Matrix) -> (Matrix, Matrix) {
    let delta = Matrix::random(first.rows, first.cols, None);
    (
        first.add_mod(&delta, RING_MODULUS),
        second.sub_mod(&delta, RING_MODULUS),
    )
}

pub fn freivalds_challenge(
    weight: &Matrix,
    r: &[i64],
) -> Result<FreivaldsChallenge, ArithmeticError> {
    Ok(FreivaldsChallenge {
        r: r.to_vec(),
        v: weight.matvec(r, PRIME_MODULUS)?,
        prime: PRIME_MODULUS,
    })
}

fn modular_dot(left: &[i64], right: &[i64], modulus: i128) -> Result<i128, ArithmeticError> {
    if left.len() != right.len() {
        return Err(ArithmeticError::Dimension);
    }
    let mut sum = 0i128;
    for (a, b) in left.iter().zip(right) {
        let product = i128::from(*a)
            .checked_mul(i128::from(*b))
            .ok_or(ArithmeticError::Overflow)?;
        sum = sum.checked_add(product).ok_or(ArithmeticError::Overflow)?;
    }
    Ok(sum.rem_euclid(modulus))
}

pub fn verify_freivalds(
    activation: &[i64],
    product: &[i64],
    challenge: &FreivaldsChallenge,
) -> Result<bool, ArithmeticError> {
    let left = modular_dot(activation, &challenge.v, challenge.prime)?;
    let right = modular_dot(product, &challenge.r, challenge.prime)?;
    Ok(left == right)
}

pub fn execute_grid(
    activation_shares: &[Vec<i64>],
    weight_shares: &[WeightShare],
    replica_fraction: f64,
) -> Result<Vec<i64>, RoleExecutionError> {
    if !(0.0..=1.0).contains(&replica_fraction) {
        return Err(RoleExecutionError {
            activation_index: 0,
            weight_index: 0,
            provider_id: "client".into(),
            reason: "invalid replica fraction".into(),
        });
    }
    let mut rng = rand::thread_rng();
    let mut results = Vec::new();
    for (activation_index, activation) in activation_shares.iter().enumerate() {
        for share in weight_shares {
            let product = share
                .matrix
                .transpose_matvec(activation, RING_MODULUS)
                .map_err(|error| RoleExecutionError {
                    activation_index,
                    weight_index: share.weight_index,
                    provider_id: share.provider_id.clone(),
                    reason: error.to_string(),
                })?;
            let verified = share
                .matrix
                .transpose_matvec(activation, share.challenge.prime)
                .map_err(|error| RoleExecutionError {
                    activation_index,
                    weight_index: share.weight_index,
                    provider_id: share.provider_id.clone(),
                    reason: error.to_string(),
                })?;
            let valid =
                verify_freivalds(activation, &verified, &share.challenge).map_err(|error| {
                    RoleExecutionError {
                        activation_index,
                        weight_index: share.weight_index,
                        provider_id: share.provider_id.clone(),
                        reason: error.to_string(),
                    }
                })?;
            if !valid {
                return Err(RoleExecutionError {
                    activation_index,
                    weight_index: share.weight_index,
                    provider_id: share.provider_id.clone(),
                    reason: "Freivalds check failed".into(),
                });
            }
            if rng.gen_bool(replica_fraction) {
                let replica = share
                    .matrix
                    .transpose_matvec(activation, RING_MODULUS)
                    .map_err(|error| RoleExecutionError {
                        activation_index,
                        weight_index: share.weight_index,
                        provider_id: share
                            .replica_provider_id
                            .clone()
                            .unwrap_or_else(|| share.provider_id.clone()),
                        reason: error.to_string(),
                    })?;
                if replica != product {
                    return Err(RoleExecutionError {
                        activation_index,
                        weight_index: share.weight_index,
                        provider_id: share
                            .replica_provider_id
                            .clone()
                            .unwrap_or_else(|| share.provider_id.clone()),
                        reason: "replica result does not match".into(),
                    });
                }
            }
            results.push(product);
        }
    }
    if results.is_empty() {
        return Err(RoleExecutionError {
            activation_index: 0,
            weight_index: 0,
            provider_id: "client".into(),
            reason: "share grid is empty".into(),
        });
    }
    let cols = results[0].len();
    Ok((0..cols)
        .map(|col| {
            results
                .iter()
                .map(|result| i128::from(result[col]))
                .sum::<i128>()
                .rem_euclid(RING_MODULUS) as i64
        })
        .collect())
}
