#![no_main]

use libfuzzer_sys::fuzz_target;
use xurcroh_crypto::{AeadSession, Sealed};

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }
    let sequence = u64::from_le_bytes(data[..8].try_into().expect("fixed prefix"));
    let nonce: [u8; 12] = data[8..20].try_into().expect("fixed nonce");
    let mut session = AeadSession::new(&[7u8; 32], b"client", b"server", b"fuzz")
        .expect("static session must initialize");
    let sealed = Sealed {
        sequence,
        nonce,
        ciphertext: data[20..].to_vec(),
    };
    let _ = session.open(b"fuzz-aad", &sealed);
});
