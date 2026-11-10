//! Authenticated QUIC tensor transport for Xurcroh.

use std::{fmt::Display, net::SocketAddr, sync::Arc};

use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::{
    pki_types::{CertificateDer, PrivatePkcs8KeyDer},
    RootCertStore,
};
use thiserror::Error;
use xurcroh_protocol::{TensorFrame, HEADER_LEN, MAX_PAYLOAD_LEN};

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
    Arc::new(rustls::crypto::ring::default_provider())
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
