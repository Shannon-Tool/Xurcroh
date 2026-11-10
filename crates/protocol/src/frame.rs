use thiserror::Error;

use crate::FORMAT_VERSION;

pub const HEADER_LEN: usize = 78;
pub const MAX_PAYLOAD_LEN: usize = 64 * 1024 * 1024;
const MAGIC: &[u8; 4] = b"XUR1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    Float64,
    Float32,
    Float16,
    BFloat16,
    Float8E4M3,
    Int64,
    Int32,
    UInt8,
}

impl DataType {
    pub const fn element_size(self) -> usize {
        match self {
            Self::Float64 | Self::Int64 => 8,
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float8E4M3 | Self::UInt8 => 1,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Float64 => 1,
            Self::Float32 => 2,
            Self::Float16 => 3,
            Self::BFloat16 => 4,
            Self::Float8E4M3 => 5,
            Self::Int64 => 6,
            Self::Int32 => 7,
            Self::UInt8 => 8,
        }
    }

    fn from_code(code: u8) -> Result<Self, FrameError> {
        match code {
            1 => Ok(Self::Float64),
            2 => Ok(Self::Float32),
            3 => Ok(Self::Float16),
            4 => Ok(Self::BFloat16),
            5 => Ok(Self::Float8E4M3),
            6 => Ok(Self::Int64),
            7 => Ok(Self::Int32),
            8 => Ok(Self::UInt8),
            _ => Err(FrameError::InvalidDataType(code)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameKind {
    Tensor,
    Cancel,
    Complete,
    Heartbeat,
    Error,
}

impl FrameKind {
    fn code(self) -> u8 {
        match self {
            Self::Tensor => 1,
            Self::Cancel => 2,
            Self::Complete => 3,
            Self::Heartbeat => 4,
            Self::Error => 5,
        }
    }

    fn from_code(code: u8) -> Result<Self, FrameError> {
        match code {
            1 => Ok(Self::Tensor),
            2 => Ok(Self::Cancel),
            3 => Ok(Self::Complete),
            4 => Ok(Self::Heartbeat),
            5 => Ok(Self::Error),
            _ => Err(FrameError::InvalidKind(code)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorFrame {
    pub request_id: [u8; 16],
    pub stream_id: u16,
    pub sequence: u64,
    pub kind: FrameKind,
    pub dtype: Option<DataType>,
    pub shape: Vec<u32>,
    pub payload: Vec<u8>,
}

impl TensorFrame {
    pub fn new(
        request_id: [u8; 16],
        stream_id: u16,
        sequence: u64,
        kind: FrameKind,
        dtype: Option<DataType>,
        shape: Vec<u32>,
        payload: Vec<u8>,
    ) -> Result<Self, FrameError> {
        let frame = Self {
            request_id,
            stream_id,
            sequence,
            kind,
            dtype,
            shape,
            payload,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<(), FrameError> {
        if self.kind == FrameKind::Tensor {
            let dtype = self.dtype.ok_or(FrameError::MissingDataType)?;
            if self.shape.len() > 8 || self.shape.contains(&0) {
                return Err(FrameError::InvalidShape);
            }
            let elements = self.shape.iter().try_fold(1u64, |product, dimension| {
                product
                    .checked_mul(u64::from(*dimension))
                    .ok_or(FrameError::InvalidShape)
            })?;
            let expected = elements
                .checked_mul(dtype.element_size() as u64)
                .ok_or(FrameError::InvalidShape)?;
            if expected != self.payload.len() as u64 {
                return Err(FrameError::PayloadLength);
            }
        } else if self.dtype.is_some() || !self.shape.is_empty() || !self.payload.is_empty() {
            return Err(FrameError::InvalidControlFrame);
        }
        if self.payload.len() > MAX_PAYLOAD_LEN {
            return Err(FrameError::PayloadTooLarge);
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, FrameError> {
        self.validate()?;
        let mut output = Vec::with_capacity(HEADER_LEN + self.payload.len());
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        output.push(self.kind.code());
        output.push(self.shape.len() as u8);
        output.extend_from_slice(&self.request_id);
        output.extend_from_slice(&self.stream_id.to_le_bytes());
        output.extend_from_slice(&self.sequence.to_le_bytes());
        output.push(self.dtype.map_or(0, DataType::code));
        output.extend_from_slice(&[0, 0, 0]);
        for index in 0..8 {
            let dimension = self.shape.get(index).copied().unwrap_or(0);
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        output.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&self.payload);
        Ok(output)
    }

    pub fn decode(input: &[u8]) -> Result<Self, FrameError> {
        if input.len() < HEADER_LEN {
            return Err(FrameError::Truncated);
        }
        if &input[..4] != MAGIC {
            return Err(FrameError::InvalidMagic);
        }
        let version =
            u16::from_le_bytes(input[4..6].try_into().map_err(|_| FrameError::Truncated)?);
        if version != FORMAT_VERSION {
            return Err(FrameError::UnsupportedVersion(version));
        }
        let kind = FrameKind::from_code(input[6])?;
        let rank = input[7] as usize;
        if rank > 8 || input[35..38] != [0, 0, 0] {
            return Err(FrameError::InvalidShape);
        }
        let mut request_id = [0u8; 16];
        request_id.copy_from_slice(&input[8..24]);
        let stream_id = u16::from_le_bytes(
            input[24..26]
                .try_into()
                .map_err(|_| FrameError::Truncated)?,
        );
        let sequence = u64::from_le_bytes(
            input[26..34]
                .try_into()
                .map_err(|_| FrameError::Truncated)?,
        );
        let dtype = match input[34] {
            0 => None,
            code => Some(DataType::from_code(code)?),
        };
        let mut shape = Vec::with_capacity(rank);
        for index in 0..rank {
            let start = 38 + index * 4;
            shape.push(u32::from_le_bytes(
                input[start..start + 4]
                    .try_into()
                    .map_err(|_| FrameError::Truncated)?,
            ));
        }
        let payload_length = u64::from_le_bytes(
            input[70..78]
                .try_into()
                .map_err(|_| FrameError::Truncated)?,
        ) as usize;
        if payload_length > MAX_PAYLOAD_LEN {
            return Err(FrameError::PayloadTooLarge);
        }
        if input.len() != HEADER_LEN + payload_length {
            return Err(FrameError::PayloadLength);
        }
        Self::new(
            request_id,
            stream_id,
            sequence,
            kind,
            dtype,
            shape,
            input[HEADER_LEN..].to_vec(),
        )
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum FrameError {
    #[error("truncated tensor frame")]
    Truncated,
    #[error("invalid tensor frame magic")]
    InvalidMagic,
    #[error("unsupported frame version {0}")]
    UnsupportedVersion(u16),
    #[error("invalid frame kind {0}")]
    InvalidKind(u8),
    #[error("invalid tensor data type {0}")]
    InvalidDataType(u8),
    #[error("tensor data type is required")]
    MissingDataType,
    #[error("invalid tensor shape")]
    InvalidShape,
    #[error("tensor payload length does not match descriptor")]
    PayloadLength,
    #[error("tensor payload exceeds the 64 MiB limit")]
    PayloadTooLarge,
    #[error("control frame carries tensor fields")]
    InvalidControlFrame,
}
