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

use crate::{encoder, iter::*, Edge};

pub fn edgelist_to_csr(edgelist: &mut [Edge]) -> (Vec<usize>, usize, Vec<u8>)
{
    edgelist.sort_unstable();

    let mut vertices =
        Vec::with_capacity(edgelist.last().map(|[u, _]| *u as usize).unwrap_or(0usize));
    let mut num_edges = 0usize;
    let mut edges = Vec::with_capacity(edgelist.len());

    for mut iter in edgelist
        .group_by(|[u, _], [v, _]| u == v)
        .map(|slice| slice.iter().dedup())
    {
        if let Some(&[u, v]) = iter.next() {
            vertices.resize(u as usize + 1, edges.len());
            num_edges += 1;
            encoder::encode(
                &mut edges,
                u,
                std::iter::once(v).chain(iter.map(|[_, w]| {
                    num_edges += 1;
                    *w
                })),
            );
        }
    }

    vertices.push(edges.len());

    vertices.shrink_to_fit();
    edges.shrink_to_fit();

    (vertices, num_edges, edges)
}
