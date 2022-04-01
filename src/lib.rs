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

#![feature(portable_simd)]
#![feature(unwrap_infallible)]
#![feature(total_cmp)]
#![feature(never_type)]
#![feature(int_log)]
#![feature(slice_as_chunks)]
#![feature(slice_split_at_unchecked)]

pub mod csr;
mod decoder;
mod edge;
mod encoder;
pub mod error;
mod iter;
mod slice;
mod vec;

pub use csr::CSR;
pub use edge::Edge;
