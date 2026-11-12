use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305,
};
use x25519_dalek::{PublicKey, StaticSecret};
use xurcroh_crypto::{seal_onion, unwrap_onion, OnionHop};

#[test]
fn onion_hops_expose_only_their_routing_layer_and_return_secret_is_innermost() {
    let secrets = [StaticSecret::from([1u8; 32]), StaticSecret::from([2u8; 32])];
    let public_a = PublicKey::from(&secrets[0]);
    let public_b = PublicKey::from(&secrets[1]);
    let return_key = [7u8; 32];
    let packet = seal_onion(
        &[
            OnionHop {
                public: public_a,
                next_hop: "worker-b".into(),
            },
            OnionHop {
                public: public_b,
                next_hop: "client-return".into(),
            },
        ],
        return_key,
        99,
        b"payload".to_vec(),
    )
    .unwrap();

    let (first, remaining) = unwrap_onion(&secrets[0], &packet.layers[0]).unwrap();
    assert_eq!(first.next_hop, "worker-b");
    assert!(first.return_key.is_none());
    let (second, final_payload) = unwrap_onion(&secrets[1], &remaining).unwrap();
    assert_eq!(second.next_hop, "client-return");
    assert_eq!(second.return_key, Some(return_key));
    assert_eq!(second.reply_id, 99);
    assert_eq!(final_payload, b"payload");
}

#[test]
fn wrong_hop_key_cannot_unwrap_onion() {
    let good = StaticSecret::from([3u8; 32]);
    let bad = StaticSecret::from([4u8; 32]);
    let packet = seal_onion(
        &[OnionHop {
            public: PublicKey::from(&good),
            next_hop: "return".into(),
        }],
        [5u8; 32],
        1,
        vec![],
    )
    .unwrap();
    assert!(unwrap_onion(&bad, &packet.layers[0]).is_err());
}

#[test]
fn onion_rejects_raw_diffie_hellman_key() {
    let secret = StaticSecret::from([8u8; 32]);
    let packet = seal_onion(
        &[OnionHop {
            public: PublicKey::from(&secret),
            next_hop: "return".into(),
        }],
        [9u8; 32],
        42,
        b"short payload".to_vec(),
    )
    .unwrap();
    let layer = &packet.layers[0];
    let ephemeral = PublicKey::from(<[u8; 32]>::try_from(&layer[..32]).unwrap());
    let shared = secret.diffie_hellman(&ephemeral);
    let raw_cipher = ChaCha20Poly1305::new(shared.as_bytes().into());
    let raw_plaintext = raw_cipher.decrypt(
        (&layer[40..52]).into(),
        Payload {
            msg: &layer[52..],
            aad: b"xurcroh-onion-v1",
        },
    );
    assert!(raw_plaintext.is_err());
}

#[test]
fn onion_pads_payloads_to_fixed_blocks() {
    let secret = StaticSecret::from([10u8; 32]);
    let packet = seal_onion(
        &[OnionHop {
            public: PublicKey::from(&secret),
            next_hop: "return".into(),
        }],
        [11u8; 32],
        43,
        b"short payload".to_vec(),
    )
    .unwrap();
    assert_eq!((packet.layers[0].len() - 52 - 16) % 1024, 0);
}
