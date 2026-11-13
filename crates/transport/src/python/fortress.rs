use pyo3::{exceptions::PyValueError, prelude::*};
use xurcroh_fortress::{execute_grid, verify_freivalds, FreivaldsChallenge, Matrix, WeightShare};

fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass]
pub struct FortressWeightShare {
    inner: WeightShare,
}

#[pymethods]
impl FortressWeightShare {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        weight_index: usize,
        rows: usize,
        cols: usize,
        data: Vec<i64>,
        r: Vec<i64>,
        v: Vec<i64>,
        prime: i128,
        provider_id: String,
        replica_provider_id: Option<String>,
    ) -> PyResult<Self> {
        let matrix = Matrix::new(rows, cols, data).map_err(value_error)?;
        Ok(Self {
            inner: WeightShare {
                weight_index,
                matrix,
                challenge: FreivaldsChallenge { r, v, prime },
                provider_id,
                replica_provider_id,
            },
        })
    }
}

#[pyfunction]
fn execute_fortress_grid(
    activation_shares: Vec<Vec<i64>>,
    weight_shares: Vec<PyRef<FortressWeightShare>>,
    replica_fraction: f64,
) -> PyResult<Vec<i64>> {
    let shares = weight_shares
        .into_iter()
        .map(|share| share.inner.clone())
        .collect::<Vec<_>>();
    execute_grid(&activation_shares, &shares, replica_fraction).map_err(value_error)
}

#[pyfunction]
fn execute_fortress_role(
    activation: Vec<i64>,
    weight_share: PyRef<FortressWeightShare>,
) -> PyResult<Vec<i64>> {
    weight_share
        .inner
        .matrix
        .transpose_matvec(&activation, xurcroh_fortress::PRIME_MODULUS)
        .map_err(value_error)
}

#[pyfunction]
fn fortress_challenge_values(
    rows: usize,
    cols: usize,
    data: &[u8],
    r: Vec<i64>,
    prime: i128,
) -> PyResult<Vec<i64>> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| value_error("matrix dimensions overflow"))?;
    let expected = elements
        .checked_mul(8)
        .ok_or_else(|| value_error("matrix byte length overflow"))?;
    if data.len() != expected || prime <= 1 {
        return Err(value_error("invalid packed matrix or prime modulus"));
    }
    let values = data
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect::<Vec<_>>();
    Matrix::new(rows, cols, values)
        .and_then(|matrix| matrix.matvec(&r, prime))
        .map_err(value_error)
}

#[pyfunction]
fn verify_fortress_product(
    activation: Vec<i64>,
    product: Vec<i64>,
    r: Vec<i64>,
    v: Vec<i64>,
    prime: i128,
) -> PyResult<bool> {
    verify_freivalds(&activation, &product, &FreivaldsChallenge { r, v, prime })
        .map_err(value_error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FortressWeightShare>()?;
    module.add_function(wrap_pyfunction!(execute_fortress_grid, module)?)?;
    module.add_function(wrap_pyfunction!(execute_fortress_role, module)?)?;
    module.add_function(wrap_pyfunction!(fortress_challenge_values, module)?)?;
    module.add_function(wrap_pyfunction!(verify_fortress_product, module)?)?;
    Ok(())
}
