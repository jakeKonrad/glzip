// glzip is a graph compression library for graph learning systems
// Copyright (C) 2022 Jacob Konrad
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#![feature(slice_group_by)]
#![feature(portable_simd)]

pub mod csr;
mod decoder;
mod encoder;
#[cfg(any(feature = "csv", feature = "npy"))]
pub mod io;
mod iter;
mod vec;

pub type Edge = [u32; 2];

pub const BYTES_PER_EDGE: usize = std::mem::size_of::<Edge>();

const KIBIBYTE: usize = 1024;

const MEBIBYTE: usize = 1024 * KIBIBYTE;

const GIBIBYTE: usize = 1024 * MEBIBYTE;

pub(in crate) const DEFAULT_EDGES_PER_CHUNK: usize = GIBIBYTE / BYTES_PER_EDGE;
