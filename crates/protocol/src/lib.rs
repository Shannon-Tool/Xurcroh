//! Versioned constants shared by the Xurcroh data plane.

mod frame;

pub use frame::{DataType, FrameError, FrameKind, TensorFrame, HEADER_LEN, MAX_PAYLOAD_LEN};

pub const FORMAT_VERSION: u16 = 1;
pub const PROTOCOL_LABEL: &str = "xurcroh-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolVersion {
    value: u16,
}

impl ProtocolVersion {
    pub fn parse(label: &str) -> Result<Self, String> {
        if label != PROTOCOL_LABEL {
            return Err(format!("unsupported protocol version: {label}"));
        }
        Ok(Self {
            value: FORMAT_VERSION,
        })
    }

    pub const fn value(self) -> u16 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::{ProtocolVersion, FORMAT_VERSION, PROTOCOL_LABEL};

    #[test]
    fn parses_only_the_current_protocol() {
        let version = ProtocolVersion::parse(PROTOCOL_LABEL).unwrap();
        assert_eq!(version.value(), FORMAT_VERSION);
        assert!(ProtocolVersion::parse("xurcroh-v2").is_err());
    }
}
