use quinn::crypto::rustls::HandshakeData;
use rustls::{
    pki_types::{CertificateDer, PrivatePkcs8KeyDer},
    NamedGroup,
};
use xurcroh_protocol::{DataType, FrameKind, TensorFrame};
use xurcroh_quic::{generate_self_signed, TensorClient, TensorServer};

fn identity_directory() -> std::path::PathBuf {
    static NEXT_IDENTITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_IDENTITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("xurcroh-mtls-{}-{sequence}", std::process::id()))
}

fn ca_signed_identity() -> (
    CertificateDer<'static>,
    CertificateDer<'static>,
    PrivatePkcs8KeyDer<'static>,
) {
    let script = r#"
from datetime import datetime, timedelta, UTC
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import sys
root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "xurcroh-test-ca")])
now = datetime.now(UTC)
ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=30)).add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True).sign(ca_key, hashes.SHA256()))
leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])).issuer_name(ca.subject).public_key(leaf_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=30)).add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True).add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False).sign(ca_key, hashes.SHA256()))
(root / "ca.der").write_bytes(ca.public_bytes(serialization.Encoding.DER))
(root / "leaf.der").write_bytes(leaf.public_bytes(serialization.Encoding.DER))
(root / "leaf-key.der").write_bytes(leaf_key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
"#;
    let directory = identity_directory();
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&directory)
        .status()
        .unwrap();
    assert!(status.success());
    (
        std::fs::read(directory.join("ca.der")).unwrap().into(),
        std::fs::read(directory.join("leaf.der")).unwrap().into(),
        PrivatePkcs8KeyDer::from(std::fs::read(directory.join("leaf-key.der")).unwrap()),
    )
}

#[test]
fn identity_directories_are_unique_per_identity() {
    let paths: Vec<_> = (0..32).map(|_| identity_directory()).collect();
    let unique: std::collections::HashSet<_> = paths.iter().collect();
    assert_eq!(unique.len(), paths.len());
}

fn frame(sequence: u64) -> TensorFrame {
    TensorFrame::new(
        [4; 16],
        1,
        sequence,
        FrameKind::Tensor,
        Some(DataType::Float32),
        vec![2],
        vec![0, 0, 128, 63, 0, 0, 0, 64],
    )
    .unwrap()
}

#[tokio::test]
async fn quic_streams_tensor_frames_with_pinned_identity() {
    let (certificate, key) = generate_self_signed("localhost").unwrap();
    let server = TensorServer::bind("127.0.0.1:0", certificate.clone(), key)
        .await
        .unwrap();
    let address = server.local_addr().unwrap();
    let server_task = tokio::spawn(async move { server.accept_frame().await.unwrap() });

    let client = TensorClient::connect(address, "localhost", certificate)
        .await
        .unwrap();
    client.send_frame(&frame(8)).await.unwrap();
    let received = server_task.await.unwrap();
    client.close().await;
    assert_eq!(received.sequence, 8);
    assert_eq!(received.payload, frame(8).payload);
}

#[tokio::test]
async fn quic_supports_response_frames_on_worker_connections() {
    let (certificate, key) = generate_self_signed("localhost").unwrap();
    let server = TensorServer::bind("127.0.0.1:0", certificate.clone(), key)
        .await
        .unwrap();
    let address = server.local_addr().unwrap();
    let server_task = tokio::spawn(async move {
        let connection = server.accept_connection().await.unwrap();
        let received = TensorServer::receive_frame(&connection).await.unwrap();
        TensorServer::send_frame(&connection, &received)
            .await
            .unwrap();
        connection.closed().await;
    });

    let client = TensorClient::connect(address, "localhost", certificate)
        .await
        .unwrap();
    client.send_frame(&frame(9)).await.unwrap();
    let response = client.receive_frame().await.unwrap();
    client.close().await;
    server_task.await.unwrap();
    assert_eq!(response.sequence, 9);
    assert_eq!(response.payload, frame(9).payload);
}

#[tokio::test]
async fn wrong_pinned_certificate_is_rejected() {
    let (server_certificate, key) = generate_self_signed("localhost").unwrap();
    let server = TensorServer::bind("127.0.0.1:0", server_certificate, key)
        .await
        .unwrap();
    let address = server.local_addr().unwrap();
    let (wrong_certificate, _) = generate_self_signed("localhost").unwrap();
    let result = TensorClient::connect(address, "localhost", wrong_certificate).await;
    assert!(result.is_err());
    drop(server);
}

#[tokio::test]
async fn mutual_tls_negotiates_hybrid_post_quantum_key_exchange() {
    let (ca_certificate, certificate, key) = ca_signed_identity();
    let client_key = rustls::pki_types::PrivatePkcs8KeyDer::from(key.secret_pkcs8_der().to_vec());
    let server = TensorServer::bind_mutual(
        "127.0.0.1:0",
        vec![certificate.clone(), ca_certificate.clone()],
        key,
        ca_certificate.clone(),
    )
    .await
    .unwrap();
    let address = server.local_addr().unwrap();
    let server_task = tokio::spawn(async move {
        let connection = server.accept_connection().await.unwrap();
        let handshake = connection
            .handshake_data()
            .unwrap()
            .downcast::<HandshakeData>()
            .unwrap();
        assert_eq!(
            handshake.negotiated_key_exchange_group,
            NamedGroup::X25519MLKEM768
        );
        let frame = TensorServer::receive_frame(&connection).await.unwrap();
        TensorServer::send_frame(&connection, &frame).await.unwrap();
        connection.closed().await;
    });

    let client = TensorClient::connect_mutual(
        address,
        "localhost",
        ca_certificate.clone(),
        vec![certificate, ca_certificate],
        client_key,
    )
    .await
    .unwrap();
    client.send_frame(&frame(10)).await.unwrap();
    let response = client.receive_frame().await.unwrap();
    assert_eq!(response.sequence, 10);
    client.close().await;
    server_task.await.unwrap();
}

#[tokio::test]
async fn mutual_tls_rejects_untrusted_client_certificate() {
    let (ca_certificate, server_certificate, server_key) = ca_signed_identity();
    let server = TensorServer::bind_mutual(
        "127.0.0.1:0",
        vec![server_certificate.clone(), ca_certificate.clone()],
        server_key,
        ca_certificate.clone(),
    )
    .await
    .unwrap();
    let address = server.local_addr().unwrap();
    let (wrong_certificate, wrong_key) = generate_self_signed("localhost").unwrap();
    let result = TensorClient::connect_mutual(
        address,
        "localhost",
        ca_certificate,
        vec![wrong_certificate],
        wrong_key,
    )
    .await;
    assert!(result.is_err());
    drop(server);
}
