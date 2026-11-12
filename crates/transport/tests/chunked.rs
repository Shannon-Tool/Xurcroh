use xurcroh_protocol::{DataType, TensorFrame};
use xurcroh_quic::{chunk_frames, reassemble_chunks};

#[test]
fn chunked_tensors_preserve_large_payloads_and_reject_bad_sequences() {
    let payload = (0..2_500_000)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let frames = chunk_frames(
        [7; 16],
        3,
        DataType::UInt8,
        &[payload.len() as u32],
        &payload,
    )
    .unwrap();
    assert!(frames.len() > 1);
    assert_eq!(reassemble_chunks(&frames).unwrap(), payload);

    let mut duplicate = frames.clone();
    duplicate.push(frames[0].clone());
    assert!(reassemble_chunks(&duplicate).is_err());

    let mut reordered = frames;
    reordered.reverse();
    assert!(reassemble_chunks(&reordered).is_err());
}

#[test]
fn chunk_validation_rejects_wrong_shape_and_total_length() {
    let mut frame = TensorFrame::new(
        [0; 16],
        0,
        0,
        xurcroh_protocol::FrameKind::Chunk,
        Some(DataType::UInt8),
        vec![2],
        vec![1, 2],
    )
    .unwrap();
    frame.payload.push(3);
    assert!(reassemble_chunks(&[frame]).is_err());
}
