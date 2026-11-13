#![no_main]

use libfuzzer_sys::fuzz_target;
use xurcroh_protocol::TensorFrame;

fuzz_target!(|data: &[u8]| {
    if let Ok(frame) = TensorFrame::decode(data) {
        let encoded = frame.encode().expect("validated frame must encode");
        assert_eq!(encoded, data);
    }
});
