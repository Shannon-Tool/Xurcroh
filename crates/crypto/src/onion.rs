use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305,
};
use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::{session::nonce, CryptoError};

pub struct OnionHop {
    pub public: PublicKey,
    pub next_hop: String,
}

pub struct OnionPacket {
    pub layers: Vec<Vec<u8>>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct HopPayload {
    pub next_hop: String,
    pub return_key: Option<[u8; 32]>,
    pub reply_id: u64,
}

pub fn seal_onion(
    hops: &[OnionHop],
    return_key: [u8; 32],
    reply_id: u64,
    mut payload: Vec<u8>,
) -> Result<OnionPacket, CryptoError> {
    if hops.is_empty() {
        return Err(CryptoError::InvalidPacket);
    }
    let mut layers = Vec::with_capacity(hops.len());
    for (index, hop) in hops.iter().enumerate().rev() {
        let inner = HopPayload {
            next_hop: hop.next_hop.clone(),
            return_key: (index == hops.len() - 1).then_some(return_key),
            reply_id,
        };
        payload = seal_layer(hop.public, &inner, &payload)?;
        layers.push(payload.clone());
    }
    layers.reverse();
    Ok(OnionPacket { layers })
}

pub fn unwrap_onion(
    secret: &StaticSecret,
    layer: &[u8],
) -> Result<(HopPayload, Vec<u8>), CryptoError> {
    if layer.len() < 32 + 8 + 12 + 16 {
        return Err(CryptoError::InvalidPacket);
    }
    let mut public_bytes = [0u8; 32];
    public_bytes.copy_from_slice(&layer[..32]);
    let ephemeral_public = PublicKey::from(public_bytes);
    let shared = secret.diffie_hellman(&ephemeral_public);
    let sequence = u64::from_le_bytes(
        layer[32..40]
            .try_into()
            .map_err(|_| CryptoError::InvalidPacket)?,
    );
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes.copy_from_slice(&layer[40..52]);
    if nonce(sequence) != nonce_bytes.into() {
        return Err(CryptoError::InvalidPacket);
    }
    let cipher = ChaCha20Poly1305::new(shared.as_bytes().into());
    let plaintext = cipher
        .decrypt(
            &nonce_bytes.into(),
            Payload {
                msg: &layer[52..],
                aad: b"xurcroh-onion-v1",
            },
        )
        .map_err(|_| CryptoError::Authentication)?;
    decode_hop(&plaintext)
}

fn seal_layer(
    public: PublicKey,
    hop: &HopPayload,
    remaining: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let ephemeral = StaticSecret::random_from_rng(OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral);
    let shared = ephemeral.diffie_hellman(&public);
    let sequence = 0u64;
    let nonce = nonce(sequence);
    let plaintext = encode_hop(hop, remaining)?;
    let cipher = ChaCha20Poly1305::new(shared.as_bytes().into());
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: &plaintext,
                aad: b"xurcroh-onion-v1",
            },
        )
        .map_err(|_| CryptoError::Authentication)?;
    let mut output = Vec::with_capacity(52 + ciphertext.len());
    output.extend_from_slice(ephemeral_public.as_bytes());
    output.extend_from_slice(&sequence.to_le_bytes());
    output.extend_from_slice(&nonce);
    output.extend_from_slice(&ciphertext);
    Ok(output)
}

fn encode_hop(hop: &HopPayload, remaining: &[u8]) -> Result<Vec<u8>, CryptoError> {
    let hop_bytes = hop.next_hop.as_bytes();
    let hop_length = u16::try_from(hop_bytes.len()).map_err(|_| CryptoError::InvalidPacket)?;
    let mut output = Vec::with_capacity(2 + hop_bytes.len() + 8 + 1 + 32 + remaining.len());
    output.extend_from_slice(&hop_length.to_le_bytes());
    output.extend_from_slice(hop_bytes);
    output.extend_from_slice(&hop.reply_id.to_le_bytes());
    output.push(u8::from(hop.return_key.is_some()));
    if let Some(key) = hop.return_key {
        output.extend_from_slice(&key);
    }
    output.extend_from_slice(remaining);
    Ok(output)
}

fn decode_hop(input: &[u8]) -> Result<(HopPayload, Vec<u8>), CryptoError> {
    if input.len() < 2 + 8 + 1 {
        return Err(CryptoError::InvalidPacket);
    }
    let hop_length = u16::from_le_bytes(
        input[..2]
            .try_into()
            .map_err(|_| CryptoError::InvalidPacket)?,
    ) as usize;
    if input.len() < 2 + hop_length + 9 {
        return Err(CryptoError::InvalidPacket);
    }
    let next_hop = String::from_utf8(input[2..2 + hop_length].to_vec())
        .map_err(|_| CryptoError::InvalidPacket)?;
    let reply_offset = 2 + hop_length;
    let reply_id = u64::from_le_bytes(
        input[reply_offset..reply_offset + 8]
            .try_into()
            .map_err(|_| CryptoError::InvalidPacket)?,
    );
    let has_return_key = input[reply_offset + 8] != 0;
    let mut offset = reply_offset + 9;
    let return_key = if has_return_key {
        if input.len() < offset + 32 {
            return Err(CryptoError::InvalidPacket);
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&input[offset..offset + 32]);
        offset += 32;
        Some(key)
    } else {
        None
    };
    Ok((
        HopPayload {
            next_hop,
            return_key,
            reply_id,
        },
        input[offset..].to_vec(),
    ))
}
