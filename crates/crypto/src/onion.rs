use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305,
};
use hkdf::Hkdf;
use rand::rngs::OsRng;
use sha2::Sha256;
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::Zeroizing;

use crate::{session::nonce, CryptoError};

const ONION_CONTEXT: &[u8] = b"xurcroh-onion-v2";
const PAYLOAD_BLOCK_LEN: usize = 1024;
const MAX_PAYLOAD_LEN: usize = 64 * 1024 * 1024;

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

fn hop_context(ephemeral: &PublicKey, recipient: &PublicKey) -> Vec<u8> {
    let mut context = Vec::with_capacity(ONION_CONTEXT.len() + 64);
    context.extend_from_slice(ONION_CONTEXT);
    context.extend_from_slice(ephemeral.as_bytes());
    context.extend_from_slice(recipient.as_bytes());
    context
}

fn hop_cipher(shared: &[u8; 32], context: &[u8]) -> Result<ChaCha20Poly1305, CryptoError> {
    let mut key = Zeroizing::new([0u8; 32]);
    Hkdf::<Sha256>::new(Some(ONION_CONTEXT), shared)
        .expand(context, &mut *key)
        .map_err(|_| CryptoError::KeyDerivation)?;
    Ok(ChaCha20Poly1305::new((&*key).into()))
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
    let context = hop_context(&ephemeral_public, &PublicKey::from(secret));
    let cipher = hop_cipher(shared.as_bytes(), &context)?;
    let plaintext = cipher
        .decrypt(
            &nonce_bytes.into(),
            Payload {
                msg: &layer[52..],
                aad: &context,
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
    let context = hop_context(&ephemeral_public, &public);
    let cipher = hop_cipher(shared.as_bytes(), &context)?;
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: &plaintext,
                aad: &context,
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
    if remaining.len() > MAX_PAYLOAD_LEN {
        return Err(CryptoError::InvalidPacket);
    }
    let hop_bytes = hop.next_hop.as_bytes();
    let hop_length = u16::try_from(hop_bytes.len()).map_err(|_| CryptoError::InvalidPacket)?;
    let return_key_len = if hop.return_key.is_some() { 32 } else { 0 };
    let plaintext_len = 2 + hop_bytes.len() + 9 + return_key_len + 8 + remaining.len();
    let padded_len = plaintext_len.div_ceil(PAYLOAD_BLOCK_LEN) * PAYLOAD_BLOCK_LEN;
    let mut output = Vec::with_capacity(padded_len);
    output.extend_from_slice(&hop_length.to_le_bytes());
    output.extend_from_slice(hop_bytes);
    output.extend_from_slice(&hop.reply_id.to_le_bytes());
    output.push(u8::from(hop.return_key.is_some()));
    if let Some(key) = hop.return_key {
        output.extend_from_slice(&key);
    }
    output.extend_from_slice(&(remaining.len() as u64).to_le_bytes());
    output.extend_from_slice(remaining);
    output.push(0x80);
    output.resize(padded_len, 0);
    Ok(output)
}

fn decode_hop(input: &[u8]) -> Result<(HopPayload, Vec<u8>), CryptoError> {
    if input.is_empty() || input.len() % PAYLOAD_BLOCK_LEN != 0 || input.len() < 2 + 8 + 1 {
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
    let return_key_flag = input[reply_offset + 8];
    if return_key_flag > 1 {
        return Err(CryptoError::InvalidPacket);
    }
    let mut offset = reply_offset + 9;
    let return_key = if return_key_flag == 1 {
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
    if input.len() < offset + 8 {
        return Err(CryptoError::InvalidPacket);
    }
    let payload_len = u64::from_le_bytes(
        input[offset..offset + 8]
            .try_into()
            .map_err(|_| CryptoError::InvalidPacket)?,
    );
    let payload_len = usize::try_from(payload_len).map_err(|_| CryptoError::InvalidPacket)?;
    if payload_len > MAX_PAYLOAD_LEN {
        return Err(CryptoError::InvalidPacket);
    }
    let payload_start = offset + 8;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or(CryptoError::InvalidPacket)?;
    if payload_end >= input.len()
        || input[payload_end] != 0x80
        || input[payload_end + 1..].iter().any(|byte| *byte != 0)
    {
        return Err(CryptoError::InvalidPacket);
    }
    Ok((
        HopPayload {
            next_hop,
            return_key,
            reply_id,
        },
        input[payload_start..payload_end].to_vec(),
    ))
}
