use xurcroh_crypto::{AeadSession, CryptoError, ReplayWindow};

fn sessions() -> (AeadSession, AeadSession) {
    let master = [9u8; 32];
    (
        AeadSession::new(&master, b"client", b"server", b"xurcroh-test").unwrap(),
        AeadSession::new(&master, b"server", b"client", b"xurcroh-test").unwrap(),
    )
}

#[test]
fn aead_round_trip_uses_directional_keys_and_associated_data() {
    let (mut client, mut server) = sessions();
    let sealed = client.seal(b"header", b"secret tensor").unwrap();
    assert_ne!(sealed.ciphertext, b"secret tensor");
    let opened = server.open(b"header", &sealed).unwrap();
    assert_eq!(opened, b"secret tensor");
}

#[test]
fn wrong_key_tampering_and_aad_are_rejected() {
    let (mut client, mut server) = sessions();
    let sealed = client.seal(b"header", b"secret").unwrap();
    assert!(server.open(b"wrong", &sealed).is_err());
    let mut modified = sealed.clone();
    modified.ciphertext[0] ^= 1;
    assert!(server.open(b"header", &modified).is_err());
    let mut other = AeadSession::new(&[8u8; 32], b"client", b"server", b"xurcroh-test").unwrap();
    assert!(other.seal(b"header", b"secret").unwrap().ciphertext != sealed.ciphertext);
}

#[test]
fn replay_is_rejected_after_allowed_out_of_order_delivery() {
    let (mut client, mut server) = sessions();
    let first = client.seal(b"header", b"first").unwrap();
    let second = client.seal(b"header", b"second").unwrap();
    assert!(server.open(b"header", &second).is_ok());
    assert!(matches!(
        server.open(b"header", &second),
        Err(CryptoError::Replay)
    ));
    assert!(server.open(b"header", &first).is_ok());
    assert!(matches!(
        server.open(b"header", &first),
        Err(CryptoError::Replay)
    ));
}

#[test]
fn replay_window_rejects_duplicate_and_too_old_sequences() {
    let mut window = ReplayWindow::new();
    assert!(window.check_and_mark(70).is_ok());
    assert!(window.check_and_mark(70).is_err());
    assert!(window.check_and_mark(1).is_err());
    assert!(window.check_and_mark(69).is_ok());
    assert!(window.check_and_mark(69).is_err());
}
