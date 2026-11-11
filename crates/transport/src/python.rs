use std::{
    fmt::Display,
    sync::{Arc, Mutex},
};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use xurcroh_crypto::CryptoError;
use xurcroh_protocol::{DataType, FrameKind, TensorFrame};

fn value_error<E: Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn crypto_error(error: CryptoError) -> PyErr {
    value_error(match error {
        CryptoError::Replay => "replay detected",
        other => return value_error(other),
    })
}

fn parse_kind(kind: &str) -> PyResult<FrameKind> {
    match kind {
        "tensor" => Ok(FrameKind::Tensor),
        "cancel" => Ok(FrameKind::Cancel),
        "complete" => Ok(FrameKind::Complete),
        "heartbeat" => Ok(FrameKind::Heartbeat),
        "error" => Ok(FrameKind::Error),
        _ => Err(value_error(format!("unknown frame kind: {kind}"))),
    }
}

fn kind_name(kind: FrameKind) -> String {
    match kind {
        FrameKind::Tensor => "tensor",
        FrameKind::Cancel => "cancel",
        FrameKind::Complete => "complete",
        FrameKind::Heartbeat => "heartbeat",
        FrameKind::Error => "error",
    }
    .into()
}

fn parse_dtype(dtype: &str) -> PyResult<DataType> {
    match dtype {
        "f64" => Ok(DataType::Float64),
        "f32" => Ok(DataType::Float32),
        "f16" => Ok(DataType::Float16),
        "bf16" => Ok(DataType::BFloat16),
        "fp8" => Ok(DataType::Float8E4M3),
        "i64" => Ok(DataType::Int64),
        "i32" => Ok(DataType::Int32),
        "u8" => Ok(DataType::UInt8),
        _ => Err(value_error(format!("unknown tensor dtype: {dtype}"))),
    }
}

fn dtype_name(dtype: DataType) -> String {
    match dtype {
        DataType::Float64 => "f64",
        DataType::Float32 => "f32",
        DataType::Float16 => "f16",
        DataType::BFloat16 => "bf16",
        DataType::Float8E4M3 => "fp8",
        DataType::Int64 => "i64",
        DataType::Int32 => "i32",
        DataType::UInt8 => "u8",
    }
    .into()
}

#[pyfunction]
fn encode_frame(
    request_id: [u8; 16],
    stream_id: u16,
    sequence: u64,
    kind: &str,
    dtype: &str,
    shape: Vec<u32>,
    payload: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let kind = parse_kind(kind)?;
    let dtype = (kind == FrameKind::Tensor)
        .then(|| parse_dtype(dtype))
        .transpose()?;
    let encoded = TensorFrame::new(request_id, stream_id, sequence, kind, dtype, shape, payload)
        .and_then(|frame| frame.encode())
        .map_err(value_error)?;
    Python::attach(|py| Ok(PyBytes::new(py, &encoded).into()))
}

#[pyfunction]
fn decode_frame(
    input: &[u8],
) -> PyResult<(
    Py<PyBytes>,
    u16,
    u64,
    String,
    Option<String>,
    Vec<u32>,
    Py<PyBytes>,
)> {
    let frame = TensorFrame::decode(input).map_err(value_error)?;
    Python::attach(|py| {
        Ok((
            PyBytes::new(py, &frame.request_id).into(),
            frame.stream_id,
            frame.sequence,
            kind_name(frame.kind),
            frame.dtype.map(dtype_name),
            frame.shape,
            PyBytes::new(py, &frame.payload).into(),
        ))
    })
}

#[pyclass]
struct AeadSession {
    inner: Mutex<xurcroh_crypto::AeadSession>,
}

#[pymethods]
impl AeadSession {
    #[new]
    fn new(
        master: &[u8],
        send_label: &[u8],
        receive_label: &[u8],
        context: &[u8],
    ) -> PyResult<Self> {
        let master: &[u8; 32] = master
            .try_into()
            .map_err(|_| value_error("master key must be 32 bytes"))?;
        let inner = xurcroh_crypto::AeadSession::new(master, send_label, receive_label, context)
            .map_err(crypto_error)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn seal(
        &self,
        associated_data: &[u8],
        plaintext: &[u8],
    ) -> PyResult<(u64, Py<PyBytes>, Py<PyBytes>)> {
        let mut session = self.inner.lock().map_err(value_error)?;
        let sealed = session
            .seal(associated_data, plaintext)
            .map_err(crypto_error)?;
        Python::attach(|py| {
            Ok((
                sealed.sequence,
                PyBytes::new(py, &sealed.nonce).into(),
                PyBytes::new(py, &sealed.ciphertext).into(),
            ))
        })
    }

    fn open(
        &self,
        associated_data: &[u8],
        sequence: u64,
        nonce: [u8; 12],
        ciphertext: &[u8],
    ) -> PyResult<Py<PyBytes>> {
        let mut session = self.inner.lock().map_err(value_error)?;
        let sealed = xurcroh_crypto::Sealed {
            sequence,
            nonce,
            ciphertext: ciphertext.to_vec(),
        };
        let plaintext = session
            .open(associated_data, &sealed)
            .map_err(crypto_error)?;
        Python::attach(|py| Ok(PyBytes::new(py, &plaintext).into()))
    }
}

use quinn::Connection;
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};
use std::{net::SocketAddr, sync::OnceLock};
use tokio::runtime::Runtime;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("Tokio runtime initialization failed"))
}

fn make_frame(
    request_id: [u8; 16],
    stream_id: u16,
    sequence: u64,
    kind: &str,
    dtype: &str,
    shape: Vec<u32>,
    payload: Vec<u8>,
) -> PyResult<TensorFrame> {
    let kind = parse_kind(kind)?;
    let dtype = (kind == FrameKind::Tensor)
        .then(|| parse_dtype(dtype))
        .transpose()?;
    TensorFrame::new(request_id, stream_id, sequence, kind, dtype, shape, payload)
        .map_err(value_error)
}

fn frame_result(
    frame: TensorFrame,
) -> PyResult<(
    Py<PyBytes>,
    u16,
    u64,
    String,
    Option<String>,
    Vec<u32>,
    Py<PyBytes>,
)> {
    Python::attach(|py| {
        Ok((
            PyBytes::new(py, &frame.request_id).into(),
            frame.stream_id,
            frame.sequence,
            kind_name(frame.kind),
            frame.dtype.map(dtype_name),
            frame.shape,
            PyBytes::new(py, &frame.payload).into(),
        ))
    })
}

#[pyfunction]
fn generate_self_signed(name: &str) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
    let (certificate, key) = crate::generate_self_signed(name).map_err(value_error)?;
    Python::attach(|py| {
        Ok((
            PyBytes::new(py, &certificate).into(),
            PyBytes::new(py, key.secret_pkcs8_der()).into(),
        ))
    })
}

#[pyclass]
struct ServerConnection {
    connection: Connection,
}

#[pymethods]
impl ServerConnection {
    fn receive(
        &self,
    ) -> PyResult<(
        Py<PyBytes>,
        u16,
        u64,
        String,
        Option<String>,
        Vec<u32>,
        Py<PyBytes>,
    )> {
        let connection = self.connection.clone();
        let frame = Python::attach(|py| {
            py.detach(|| runtime().block_on(crate::TensorServer::receive_frame(&connection)))
        })
        .map_err(value_error)?;
        frame_result(frame)
    }

    fn send(
        &self,
        request_id: [u8; 16],
        stream_id: u16,
        sequence: u64,
        kind: &str,
        dtype: &str,
        shape: Vec<u32>,
        payload: Vec<u8>,
    ) -> PyResult<()> {
        let frame = make_frame(request_id, stream_id, sequence, kind, dtype, shape, payload)?;
        let connection = self.connection.clone();
        Python::attach(|py| {
            py.detach(|| runtime().block_on(crate::TensorServer::send_frame(&connection, &frame)))
        })
        .map_err(value_error)
    }

    fn close(&self) {
        let connection = self.connection.clone();
        Python::attach(|py| py.detach(|| runtime().block_on(connection.closed())));
    }
}

#[pyclass]
struct TensorServer {
    inner: Mutex<Option<Arc<crate::TensorServer>>>,
}

#[pymethods]
impl TensorServer {
    #[new]
    fn bind(address: &str, certificate: Vec<u8>, key: Vec<u8>) -> PyResult<Self> {
        let certificate = CertificateDer::from(certificate);
        let key = PrivatePkcs8KeyDer::from(key);
        let server = Python::attach(|py| {
            py.detach(|| runtime().block_on(crate::TensorServer::bind(address, certificate, key)))
        })
        .map_err(value_error)?;
        Ok(Self {
            inner: Mutex::new(Some(Arc::new(server))),
        })
    }

    fn local_addr(&self) -> PyResult<String> {
        let server = self.inner.lock().map_err(value_error)?;
        server
            .as_ref()
            .ok_or_else(|| value_error("server is closed"))?
            .local_addr()
            .map(|address| address.to_string())
            .map_err(value_error)
    }

    fn accept(&self) -> PyResult<ServerConnection> {
        let server = {
            let guard = self.inner.lock().map_err(value_error)?;
            guard
                .as_ref()
                .ok_or_else(|| value_error("server is closed"))?
                .clone()
        };
        let connection =
            Python::attach(|py| py.detach(|| runtime().block_on(server.accept_connection())))
                .map_err(value_error)?;
        Ok(ServerConnection { connection })
    }

    fn close(&self) -> PyResult<()> {
        let server = self.inner.lock().map_err(value_error)?.take();
        if let Some(server) = server {
            server.close();
        }
        Ok(())
    }
}

#[pyclass]
struct TensorClient {
    inner: Mutex<Option<Arc<crate::TensorClient>>>,
}

#[pymethods]
impl TensorClient {
    #[new]
    fn connect(address: &str, server_name: &str, certificate: Vec<u8>) -> PyResult<Self> {
        let address: SocketAddr = address.parse().map_err(value_error)?;
        let certificate = CertificateDer::from(certificate);
        let client = Python::attach(|py| {
            py.detach(|| {
                runtime().block_on(crate::TensorClient::connect(
                    address,
                    server_name,
                    certificate,
                ))
            })
        })
        .map_err(value_error)?;
        Ok(Self {
            inner: Mutex::new(Some(Arc::new(client))),
        })
    }

    fn send(
        &self,
        request_id: [u8; 16],
        stream_id: u16,
        sequence: u64,
        kind: &str,
        dtype: &str,
        shape: Vec<u32>,
        payload: Vec<u8>,
    ) -> PyResult<()> {
        let frame = make_frame(request_id, stream_id, sequence, kind, dtype, shape, payload)?;
        let client = {
            let guard = self.inner.lock().map_err(value_error)?;
            guard
                .as_ref()
                .ok_or_else(|| value_error("client is closed"))?
                .clone()
        };
        Python::attach(|py| py.detach(|| runtime().block_on(client.send_frame(&frame))))
            .map_err(value_error)
    }

    fn receive(
        &self,
    ) -> PyResult<(
        Py<PyBytes>,
        u16,
        u64,
        String,
        Option<String>,
        Vec<u32>,
        Py<PyBytes>,
    )> {
        let client = {
            let guard = self.inner.lock().map_err(value_error)?;
            guard
                .as_ref()
                .ok_or_else(|| value_error("client is closed"))?
                .clone()
        };
        let frame = Python::attach(|py| py.detach(|| runtime().block_on(client.receive_frame())))
            .map_err(value_error)?;
        frame_result(frame)
    }

    fn close(&self) -> PyResult<()> {
        let client = self.inner.lock().map_err(value_error)?.take();
        if let Some(client) = client {
            client.abort();
            let client =
                Arc::try_unwrap(client).map_err(|_| value_error("client is still in use"))?;
            Python::attach(|py| py.detach(|| runtime().block_on(client.close())));
        }
        Ok(())
    }
}

#[pymodule]
fn xurcroh_quic(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AeadSession>()?;
    module.add_class::<TensorServer>()?;
    module.add_class::<TensorClient>()?;
    module.add_class::<ServerConnection>()?;
    module.add_function(wrap_pyfunction!(encode_frame, module)?)?;
    module.add_function(wrap_pyfunction!(decode_frame, module)?)?;
    module.add_function(wrap_pyfunction!(generate_self_signed, module)?)?;
    Ok(())
}
