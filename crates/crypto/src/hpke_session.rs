use hpke::{
    aead::ChaCha20Poly1305,
    kdf::HkdfSha256,
    kem::{Kem, X25519HkdfSha256},
    Deserializable, OpModeR, OpModeS, Serializable,
};

use crate::CryptoError;

pub struct HpkeEnvelope {
    pub encapsulated_key: [u8; 32],
    pub ciphertext: Vec<u8>,
}

pub fn seal_session_master(
    recipient_public: &[u8],
    info: &[u8],
    associated_data: &[u8],
    master: &[u8; 32],
) -> Result<HpkeEnvelope, CryptoError> {
    let public = <X25519HkdfSha256 as Kem>::PublicKey::from_bytes(recipient_public)
        .map_err(|_| CryptoError::InvalidPacket)?;
    let (encapsulated, ciphertext) = hpke::single_shot_seal::<
        ChaCha20Poly1305,
        HkdfSha256,
        X25519HkdfSha256,
    >(&OpModeS::Base, &public, info, master, associated_data)
    .map_err(|_| CryptoError::KeyDerivation)?;
    let encapsulated_key: [u8; 32] = encapsulated
        .to_bytes()
        .as_slice()
        .try_into()
        .map_err(|_| CryptoError::InvalidPacket)?;
    Ok(HpkeEnvelope {
        encapsulated_key,
        ciphertext,
    })
}

pub fn open_session_master(
    recipient_private: &[u8],
    encapsulated_key: &[u8],
    ciphertext: &[u8],
    info: &[u8],
    associated_data: &[u8],
) -> Result<[u8; 32], CryptoError> {
    let private = <X25519HkdfSha256 as Kem>::PrivateKey::from_bytes(recipient_private)
        .map_err(|_| CryptoError::InvalidPacket)?;
    let encapsulated = <X25519HkdfSha256 as Kem>::EncappedKey::from_bytes(encapsulated_key)
        .map_err(|_| CryptoError::InvalidPacket)?;
    let master = hpke::single_shot_open::<ChaCha20Poly1305, HkdfSha256, X25519HkdfSha256>(
        &OpModeR::Base,
        &private,
        &encapsulated,
        info,
        ciphertext,
        associated_data,
    )
    .map_err(|_| CryptoError::Authentication)?;
    master
        .as_slice()
        .try_into()
        .map_err(|_| CryptoError::InvalidPacket)
}
