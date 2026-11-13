#![no_main]

use libfuzzer_sys::fuzz_target;
use xurcroh_crypto::open_session_master;

fuzz_target!(|data: &[u8]| {
    if data.len() < 64 {
        return;
    }
    let _ = open_session_master(
        &data[..32],
        &data[32..64],
        &data[64..],
        b"xurcroh-fuzz-hpke",
        b"fuzz-session",
    );
});
