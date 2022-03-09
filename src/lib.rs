
pub mod encoder;
pub mod decoder;
pub mod csr;

pub type Edge = [u32; 2];

pub const EDGE_BYTES: usize = std::mem::size_of::<Edge>();

pub const KIBIBYTE: usize = 1024;

pub const MEBIBYTE: usize = 1024 * KIBIBYTE;

pub const GIBIBYTE: usize = 1024 * MEBIBYTE;

pub const TEBIBYTE: usize = 1024 * GIBIBYTE;

pub const PEBIBYTE: usize = 1024 * TEBIBYTE;

