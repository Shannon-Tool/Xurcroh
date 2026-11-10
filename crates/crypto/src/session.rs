use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305, Nonce,
};
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::CryptoError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sealed {
    pub sequence: u64,
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct ReplayWindow {
    highest: Option<u64>,
    bitmap: u64,
}

impl ReplayWindow {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check_and_mark(&mut self, sequence: u64) -> Result<(), CryptoError> {
        let Some(highest) = self.highest else {
            self.highest = Some(sequence);
            self.bitmap = 1;
            return Ok(());
        };
        if sequence > highest {
            let shift = sequence - highest;
            self.bitmap = if shift >= 64 {
                1
            } else {
                (self.bitmap << shift) | 1
            };
            self.highest = Some(sequence);
            return Ok(());
        }
        let distance = highest - sequence;
        if distance >= 64 {
            return Err(CryptoError::Replay);
        }
        let mask = 1u64 << distance;
        if self.bitmap & mask != 0 {
            return Err(CryptoError::Replay);
        }
        self.bitmap |= mask;
        Ok(())
    }
}

pub struct AeadSession {
    send_key: Zeroizing<[u8; 32]>,
    receive_key: Zeroizing<[u8; 32]>,
    next_send_sequence: u64,
    replay: ReplayWindow,
}

impl AeadSession {
    pub fn new(
        master: &[u8; 32],
        send_label: &[u8],
        receive_label: &[u8],
        context: &[u8],
    ) -> Result<Self, CryptoError> {
        Ok(Self {
            send_key: derive_key(master, send_label, context)?,
            receive_key: derive_key(master, receive_label, context)?,
            next_send_sequence: 0,
            replay: ReplayWindow::new(),
        })
    }

    pub fn rotate(
        &mut self,
        master: &[u8; 32],
        send_label: &[u8],
        receive_label: &[u8],
        context: &[u8],
        epoch: u64,
    ) -> Result<(), CryptoError> {
        let mut rotation_context = Vec::with_capacity(context.len() + 8);
        rotation_context.extend_from_slice(context);
        rotation_context.extend_from_slice(&epoch.to_le_bytes());
        let send_key = derive_key(master, send_label, &rotation_context)?;
        let receive_key = derive_key(master, receive_label, &rotation_context)?;
        self.send_key = send_key;
        self.receive_key = receive_key;
        self.next_send_sequence = 0;
        self.replay = ReplayWindow::new();
        Ok(())
    }

    pub fn seal(
        &mut self,
        associated_data: &[u8],
        plaintext: &[u8],
    ) -> Result<Sealed, CryptoError> {
        if self.next_send_sequence == u64::MAX {
            return Err(CryptoError::NonceExhausted);
        }
        let sequence = self.next_send_sequence;
        self.next_send_sequence = self
            .next_send_sequence
            .checked_add(1)
            .ok_or(CryptoError::NonceExhausted)?;
        let nonce = nonce(sequence);
        let cipher = ChaCha20Poly1305::new((&*self.send_key).into());
        let ciphertext = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext,
                    aad: associated_data,
                },
            )
            .map_err(|_| CryptoError::Authentication)?;
        Ok(Sealed {
            sequence,
            nonce: nonce.into(),
            ciphertext,
        })
    }

    pub fn open(
        &mut self,
        associated_data: &[u8],
        sealed: &Sealed,
    ) -> Result<Vec<u8>, CryptoError> {
        if nonce(sealed.sequence) != sealed.nonce.into() {
            return Err(CryptoError::InvalidPacket);
        }
        let cipher = ChaCha20Poly1305::new((&*self.receive_key).into());
        let plaintext = cipher
            .decrypt(
                (&sealed.nonce).into(),
                Payload {
                    msg: &sealed.ciphertext,
                    aad: associated_data,
                },
            )
            .map_err(|_| CryptoError::Authentication)?;
        self.replay.check_and_mark(sealed.sequence)?;
        Ok(plaintext)
    }
}

fn derive_key(
    master: &[u8; 32],
    label: &[u8],
    context: &[u8],
) -> Result<Zeroizing<[u8; 32]>, CryptoError> {
    let mut key = Zeroizing::new([0u8; 32]);
    Hkdf::<Sha256>::new(Some(context), master)
        .expand(label, &mut *key)
        .map_err(|_| CryptoError::KeyDerivation)?;
    Ok(key)
}

pub(crate) fn nonce(sequence: u64) -> Nonce {
    let mut bytes = [0u8; 12];
    bytes[4..].copy_from_slice(&sequence.to_le_bytes());
    Nonce::from(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_exhaustion_fails_closed() {
        let mut session = AeadSession::new(&[0u8; 32], b"a", b"b", b"test").unwrap();
        session.next_send_sequence = u64::MAX;
        assert_eq!(
            session.seal(b"header", b"payload"),
            Err(CryptoError::NonceExhausted)
        );
    }
}
