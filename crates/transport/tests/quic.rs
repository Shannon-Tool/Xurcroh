use xurcroh_protocol::{DataType, FrameKind, TensorFrame};
use xurcroh_quic::{generate_self_signed, TensorClient, TensorServer};

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
