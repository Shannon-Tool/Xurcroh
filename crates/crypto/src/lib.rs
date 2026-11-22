//! Cryptographic session primitives for Xurcroh.

mod session;

pub use session::{AeadSession, ReplayWindow, Sealed};

use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CryptoError {
    #[error("session key derivation failed")]
    KeyDerivation,
    #[error("session nonce space is exhausted")]
    NonceExhausted,
    #[error("replayed or too-old packet")]
    Replay,
    #[error("packet authentication failed")]
    Authentication,
    #[error("invalid encrypted packet")]
    InvalidPacket,
}
