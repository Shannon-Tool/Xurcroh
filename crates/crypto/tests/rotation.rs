use xurcroh_crypto::AeadSession;

#[test]
fn rotation_replaces_keys_and_starts_a_new_nonce_domain() {
    let master = [11u8; 32];
    let mut client = AeadSession::new(&master, b"client", b"server", b"rotate").unwrap();
    let mut server = AeadSession::new(&master, b"server", b"client", b"rotate").unwrap();
    let old = client.seal(b"header", b"old").unwrap();
    assert!(server.open(b"header", &old).is_ok());
    let stale = client.seal(b"header", b"stale").unwrap();

    client
        .rotate(&master, b"client", b"server", b"rotate", 1)
        .unwrap();
    server
        .rotate(&master, b"server", b"client", b"rotate", 1)
        .unwrap();
    assert!(server.open(b"header", &stale).is_err());
    let current = client.seal(b"header", b"current").unwrap();
    assert_eq!(current.sequence, 0);
    assert_eq!(server.open(b"header", &current).unwrap(), b"current");
}
