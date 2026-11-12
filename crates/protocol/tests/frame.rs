use proptest::prelude::*;
use xurcroh_protocol::{DataType, FrameKind, TensorFrame, HEADER_LEN};

fn frame() -> TensorFrame {
    TensorFrame::new(
        [7; 16],
        3,
        42,
        FrameKind::Tensor,
        Some(DataType::BFloat16),
        vec![2, 3],
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    .unwrap()
}

#[test]
fn frame_round_trip_preserves_request_stream_sequence_and_tensor() {
    let encoded = frame().encode().unwrap();
    let decoded = TensorFrame::decode(&encoded).unwrap();
    assert_eq!(decoded.request_id, [7; 16]);
    assert_eq!(decoded.stream_id, 3);
    assert_eq!(decoded.sequence, 42);
    assert_eq!(decoded.shape, vec![2, 3]);
    assert_eq!(decoded.payload, (1..=12).collect::<Vec<_>>());
}

#[test]
fn malformed_frames_fail_without_panicking() {
    let encoded = frame().encode().unwrap();
    for length in 0..encoded.len() {
        assert!(TensorFrame::decode(&encoded[..length]).is_err());
    }
    let mut wrong_shape = encoded.clone();
    wrong_shape[38..42].copy_from_slice(&99u32.to_le_bytes());
    assert!(TensorFrame::decode(&wrong_shape).is_err());
    let mut trailing = encoded;
    trailing.push(0);
    assert!(TensorFrame::decode(&trailing).is_err());
}

#[test]
fn control_frames_cannot_carry_tensor_payloads() {
    let result = TensorFrame::new([0; 16], 0, 0, FrameKind::Cancel, None, vec![], vec![1]);
    assert!(result.is_err());
}

#[test]
fn cancellation_carries_a_bounded_session_proof() {
    let frame =
        TensorFrame::new([0; 16], 0, 0, FrameKind::Cancel, None, vec![], vec![7; 32]).unwrap();
    let decoded = TensorFrame::decode(&frame.encode().unwrap()).unwrap();
    assert_eq!(decoded.payload, vec![7; 32]);
    assert!(TensorFrame::new([0; 16], 0, 0, FrameKind::Cancel, None, vec![], vec![7; 31]).is_err());
    assert!(TensorFrame::new(
        [0; 16],
        0,
        0,
        FrameKind::Complete,
        None,
        vec![],
        vec![7; 32]
    )
    .is_err());
}

proptest! {
    #[test]
    fn arbitrary_bytes_never_panic(payload in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = TensorFrame::decode(&payload);
    }

    #[test]
    fn valid_payload_size_is_bounded(overhead in 0usize..64) {
        let payload = vec![0u8; 12 + overhead];
        let shape = vec![2, 3];
        let result = TensorFrame::new(
            [0; 16], 0, 0, FrameKind::Tensor, Some(DataType::BFloat16), shape, payload,
        );
        assert_eq!(result.is_ok(), overhead == 0);
    }
}

#[test]
fn encoded_header_has_stable_size() {
    assert_eq!(HEADER_LEN, 78);
    assert!(frame().encode().unwrap().len() > HEADER_LEN);
}
