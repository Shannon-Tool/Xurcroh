//! Authenticated QUIC tensor transport for Xurcroh.

use std::{
    fmt::Display,
    net::SocketAddr,
    sync::{Arc, OnceLock},
};

use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::{
    pki_types::{CertificateDer, PrivatePkcs8KeyDer},
    RootCertStore,
};
use thiserror::Error;
use xurcroh_protocol::{DataType, FrameKind, TensorFrame, HEADER_LEN, MAX_PAYLOAD_LEN};

pub const CHUNK_PAYLOAD_LEN: usize = 1024 * 1024;
pub const MAX_STREAM_PAYLOAD_LEN: usize = 16 * 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("transport setup failed: {0}")]
    Setup(String),
    #[error("QUIC connection failed: {0}")]
    Connection(String),
    #[error("tensor protocol failed: {0}")]
    Protocol(String),
}

fn setup<E: Display>(error: E) -> TransportError {
    TransportError::Setup(error.to_string())
}

fn connection_error<E: Display>(error: E) -> TransportError {
    TransportError::Connection(error.to_string())
}

pub fn generate_self_signed(
    name: &str,
) -> Result<(CertificateDer<'static>, PrivatePkcs8KeyDer<'static>), TransportError> {
    let certified = generate_simple_self_signed(vec![name.to_owned()]).map_err(setup)?;
    Ok((
        certified.cert.der().clone(),
        PrivatePkcs8KeyDer::from(certified.key_pair.serialize_der()),
    ))
}

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    static INSTALL: OnceLock<()> = OnceLock::new();
    INSTALL.get_or_init(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
    Arc::new(rustls::crypto::aws_lc_rs::default_provider())
}

pub struct TensorServer {
    endpoint: Endpoint,
}

impl TensorServer {
    pub async fn bind(
        address: &str,
        certificate: CertificateDer<'static>,
        key: PrivatePkcs8KeyDer<'static>,
    ) -> Result<Self, TransportError> {
        let mut crypto = rustls::ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .map_err(setup)?
            .with_no_client_auth()
            .with_single_cert(vec![certificate], key.into())
            .map_err(setup)?;
        crypto.alpn_protocols = vec![b"xurcroh-v1".to_vec()];
        let config = ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(crypto).map_err(setup)?,
        ));
        let socket: SocketAddr = address.parse().map_err(setup)?;
        let endpoint = Endpoint::server(config, socket).map_err(setup)?;
        Ok(Self { endpoint })
    }

    pub async fn bind_mutual(
        address: &str,
        certificates: Vec<CertificateDer<'static>>,
        key: PrivatePkcs8KeyDer<'static>,
        client_ca: CertificateDer<'static>,
    ) -> Result<Self, TransportError> {
        let mut roots = RootCertStore::empty();
        roots.add(client_ca).map_err(setup)?;
        let verifier =
            rustls::server::WebPkiClientVerifier::builder_with_provider(roots.into(), provider())
                .build()
                .map_err(setup)?;
        let mut crypto = rustls::ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .map_err(setup)?
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates, key.into())
            .map_err(setup)?;
        crypto.alpn_protocols = vec![b"xurcroh-v1".to_vec()];
        let config = ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(crypto).map_err(setup)?,
        ));
        let socket: SocketAddr = address.parse().map_err(setup)?;
        let endpoint = Endpoint::server(config, socket).map_err(setup)?;
        Ok(Self { endpoint })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.endpoint.local_addr().map_err(setup)
    }

    pub fn close(&self) {
        self.endpoint.close(0u32.into(), b"shutdown");
    }

    pub async fn accept_connection(&self) -> Result<Connection, TransportError> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| TransportError::Connection("server endpoint closed".into()))?;
        incoming.await.map_err(connection_error)
    }

    pub async fn receive_frame(connection: &Connection) -> Result<TensorFrame, TransportError> {
        let (_, mut receive) = connection.accept_bi().await.map_err(connection_error)?;
        let mut encoded = vec![0u8; HEADER_LEN];
        receive
            .read_exact(&mut encoded)
            .await
            .map_err(connection_error)?;
        let payload_length = u64::from_le_bytes(
            encoded[70..78]
                .try_into()
                .map_err(|_| TransportError::Protocol("truncated payload length".into()))?,
        ) as usize;
        if payload_length > MAX_PAYLOAD_LEN {
            return Err(TransportError::Protocol("payload exceeds limit".into()));
        }
        encoded.resize(HEADER_LEN + payload_length, 0);
        if payload_length > 0 {
            receive
                .read_exact(&mut encoded[HEADER_LEN..])
                .await
                .map_err(connection_error)?;
        }
        TensorFrame::decode(&encoded).map_err(|error| TransportError::Protocol(error.to_string()))
    }

    pub async fn send_frame(
        connection: &Connection,
        frame: &TensorFrame,
    ) -> Result<(), TransportError> {
        let encoded = frame
            .encode()
            .map_err(|error| TransportError::Protocol(error.to_string()))?;
        let (mut send, _) = connection.open_bi().await.map_err(connection_error)?;
        send.write_all(&encoded).await.map_err(connection_error)?;
        send.finish().map_err(connection_error)?;
        Ok(())
    }

    pub async fn accept_frame(&self) -> Result<TensorFrame, TransportError> {
        let connection = self.accept_connection().await?;
        Self::receive_frame(&connection).await
    }

    pub async fn send_chunked(
        connection: &Connection,
        request_id: [u8; 16],
        stream_id: u16,
        dtype: DataType,
        shape: &[u32],
        payload: &[u8],
    ) -> Result<(), TransportError> {
        for frame in chunk_frames(request_id, stream_id, dtype, shape, payload)? {
            Self::send_frame(connection, &frame).await?;
        }
        Ok(())
    }

    pub async fn receive_chunked(connection: &Connection) -> Result<Vec<u8>, TransportError> {
        let first = Self::receive_frame(connection).await?;
        let expected = first
            .shape
            .iter()
            .try_fold(1u64, |product, dimension| {
                product.checked_mul(u64::from(*dimension))
            })
            .and_then(|elements| elements.checked_mul(first.dtype?.element_size() as u64))
            .ok_or_else(|| TransportError::Protocol("stream length overflow".into()))?
            as usize;
        let mut frames = vec![first];
        let mut received = frames[0].payload.len();
        while received < expected {
            let frame = Self::receive_frame(connection).await?;
            received = received
                .checked_add(frame.payload.len())
                .ok_or_else(|| TransportError::Protocol("stream length overflow".into()))?;
            if received > expected {
                return Err(TransportError::Protocol(
                    "stream exceeds declared length".into(),
                ));
            }
            frames.push(frame);
        }
        reassemble_chunks(&frames)
    }
}

pub fn chunk_frames(
    request_id: [u8; 16],
    stream_id: u16,
    dtype: DataType,
    shape: &[u32],
    payload: &[u8],
) -> Result<Vec<TensorFrame>, TransportError> {
    let expected = shape
        .iter()
        .try_fold(1u64, |product, dimension| {
            product.checked_mul(u64::from(*dimension))
        })
        .and_then(|elements| elements.checked_mul(dtype.element_size() as u64))
        .ok_or_else(|| TransportError::Protocol("stream length overflow".into()))?;
    if expected != payload.len() as u64 || payload.len() > MAX_STREAM_PAYLOAD_LEN {
        return Err(TransportError::Protocol(
            "invalid stream payload length".into(),
        ));
    }
    payload
        .chunks(CHUNK_PAYLOAD_LEN)
        .enumerate()
        .map(|(sequence, chunk)| {
            TensorFrame::new(
                request_id,
                stream_id,
                sequence as u64,
                FrameKind::Chunk,
                Some(dtype),
                shape.to_vec(),
                chunk.to_vec(),
            )
            .map_err(|error| TransportError::Protocol(error.to_string()))
        })
        .collect()
}

pub fn reassemble_chunks(frames: &[TensorFrame]) -> Result<Vec<u8>, TransportError> {
    let first = frames
        .first()
        .ok_or_else(|| TransportError::Protocol("empty chunk stream".into()))?;
    let dtype = first
        .dtype
        .ok_or_else(|| TransportError::Protocol("missing dtype".into()))?;
    let shape = first.shape.clone();
    let expected = shape
        .iter()
        .try_fold(1u64, |product, dimension| {
            product.checked_mul(u64::from(*dimension))
        })
        .and_then(|elements| elements.checked_mul(dtype.element_size() as u64))
        .ok_or_else(|| TransportError::Protocol("stream length overflow".into()))?
        as usize;
    let mut payload = Vec::with_capacity(expected);
    for (index, frame) in frames.iter().enumerate() {
        if frame.request_id != first.request_id
            || frame.stream_id != first.stream_id
            || frame.sequence != index as u64
            || frame.kind != FrameKind::Chunk
            || frame.dtype != Some(dtype)
            || frame.shape != shape
        {
            return Err(TransportError::Protocol(
                "inconsistent chunk sequence".into(),
            ));
        }
        if payload.len() + frame.payload.len() > expected {
            return Err(TransportError::Protocol(
                "stream exceeds declared length".into(),
            ));
        }
        payload.extend_from_slice(&frame.payload);
    }
    if payload.len() != expected {
        return Err(TransportError::Protocol("incomplete chunk stream".into()));
    }
    Ok(payload)
}

impl TensorServer {
    pub async fn accept_chunked(&self) -> Result<Vec<u8>, TransportError> {
        let connection = self.accept_connection().await?;
        Self::receive_chunked(&connection).await
    }
}

pub struct TensorClient {
    endpoint: Endpoint,
    connection: Connection,
}

impl TensorClient {
    pub async fn connect(
        address: SocketAddr,
        server_name: &str,
        pinned_certificate: CertificateDer<'static>,
    ) -> Result<Self, TransportError> {
        let mut roots = RootCertStore::empty();
        roots.add(pinned_certificate).map_err(setup)?;
        let mut crypto = rustls::ClientConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .map_err(setup)?
            .with_root_certificates(roots)
            .with_no_client_auth();
        crypto.alpn_protocols = vec![b"xurcroh-v1".to_vec()];
        let config = ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(setup)?,
        ));
        let socket: SocketAddr = "0.0.0.0:0".parse().map_err(setup)?;
        let mut endpoint = Endpoint::client(socket).map_err(setup)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint
            .connect(address, server_name)
            .map_err(connection_error)?
            .await
            .map_err(connection_error)?;
        Ok(Self {
            endpoint,
            connection,
        })
    }

    pub async fn connect_mutual(
        address: SocketAddr,
        server_name: &str,
        pinned_certificate: CertificateDer<'static>,
        client_certificates: Vec<CertificateDer<'static>>,
        client_key: PrivatePkcs8KeyDer<'static>,
    ) -> Result<Self, TransportError> {
        let mut roots = RootCertStore::empty();
        roots.add(pinned_certificate).map_err(setup)?;
        let mut crypto = rustls::ClientConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .map_err(setup)?
            .with_root_certificates(roots)
            .with_client_auth_cert(client_certificates, client_key.into())
            .map_err(setup)?;
        crypto.alpn_protocols = vec![b"xurcroh-v1".to_vec()];
        let config = ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(setup)?,
        ));
        let socket: SocketAddr = "0.0.0.0:0".parse().map_err(setup)?;
        let mut endpoint = Endpoint::client(socket).map_err(setup)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint
            .connect(address, server_name)
            .map_err(connection_error)?
            .await
            .map_err(connection_error)?;
        Ok(Self {
            endpoint,
            connection,
        })
    }

    pub async fn send_frame(&self, frame: &TensorFrame) -> Result<(), TransportError> {
        let encoded = frame
            .encode()
            .map_err(|error| TransportError::Protocol(error.to_string()))?;
        let (mut send, _) = self.connection.open_bi().await.map_err(connection_error)?;
        send.write_all(&encoded).await.map_err(connection_error)?;
        send.finish().map_err(connection_error)?;
        Ok(())
    }

    pub async fn receive_frame(&self) -> Result<TensorFrame, TransportError> {
        TensorServer::receive_frame(&self.connection).await
    }

    pub fn abort(&self) {
        self.endpoint.close(0u32.into(), b"aborted");
    }

    pub async fn close(self) {
        self.connection.close(0u32.into(), b"complete");
        self.endpoint.wait_idle().await;
    }
}

#[cfg(feature = "python")]
mod python;
