use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};
use xurcroh_crypto::{open_session_master, seal_session_master};

#[test]
fn hpke_session_master_round_trip_and_tamper_rejection() {
    let secret = StaticSecret::random_from_rng(OsRng);
    let public = PublicKey::from(&secret);
    let envelope = seal_session_master(
        public.as_bytes(),
        b"stage-hpke-test",
        b"session-and-token",
        &[9u8; 32],
    )
    .unwrap();
    let opened = open_session_master(
        secret.as_bytes(),
        &envelope.encapsulated_key,
        &envelope.ciphertext,
        b"stage-hpke-test",
        b"session-and-token",
    )
    .unwrap();
    assert_eq!(opened, [9u8; 32]);

    let mut tampered = envelope.ciphertext.clone();
    tampered[0] ^= 1;
    assert!(open_session_master(
        secret.as_bytes(),
        &envelope.encapsulated_key,
        &tampered,
        b"stage-hpke-test",
        b"session-and-token",
    )
    .is_err());
}
