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
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Compress Sparse Row representation of directed graphs but even more compressed as the
//! edges are compressed using the byte run length encoding scheme from
//! [Ligra+](https://people.csail.mit.edu/jshun/ligra+.pdf).

use rayon::prelude::*;

use crate::{decoder, encoder, iter::IteratorDedupExt, par, vec, Edge};

/// The Compressed Sparse Row struct.
pub struct CSR
{
    vertices: Vec<usize>,
    num_edges: usize,
    edges: Vec<u8>,
}

impl CSR
{
    /// The neighbors of a vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip::CSR;
    ///
    /// let csr = CSR::from(vec![
    ///     [0u32,1],
    ///     [0,2],
    ///     [1,0],
    ///     [2,1],
    /// ]);
    ///
    /// assert_eq!(vec![1,2], csr.neighbors(0).collect::<Vec<_>>());
    /// assert_eq!(vec![0], csr.neighbors(1).collect::<Vec<_>>());
    /// assert_eq!(vec![1], csr.neighbors(2).collect::<Vec<_>>());
    /// ```
    pub fn neighbors(&self, source: u32) -> impl Iterator<Item = u32> + '_
    {
        let i = source as usize;
        self.vertices.get(i).into_iter().flat_map(move |&start| {
            self.vertices
                .get(i + 1)
                .into_iter()
                .flat_map(move |&end| decoder::decode(source, &self.edges[start..end]))
        })
    }

    /// The degrees of a vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip::CSR;
    ///
    /// let csr = CSR::from(vec![
    ///     [0u32,1],
    ///     [0,2],
    ///     [1,0],
    ///     [2,1],
    /// ]);
    ///
    /// assert_eq!(2, csr.degree(0));
    /// assert_eq!(1, csr.degree(1));
    /// assert_eq!(1, csr.degree(2));
    /// ```
    pub fn degree(&self, source: u32) -> usize
    {
        let i = source as usize;
        self.vertices
            .get(i)
            .and_then(move |&start| {
                self.vertices
                    .get(i + 1)
                    .map(|&end| decoder::count(source, &self.edges[start..end]))
            })
            .unwrap_or(0usize)
    }

    /// The edges of a graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip::{CSR, Edge};
    ///
    /// let csr = CSR::from(vec![
    ///     [0u32,1],
    ///     [0,2],
    ///     [1,0],
    ///     [2,1],
    /// ]);
    ///
    /// let es = vec![Edge(0, 1), Edge(0, 2), Edge(1, 0), Edge(2, 1)];
    ///
    /// assert_eq!(es, csr.edges().collect::<Vec<_>>());
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_
    {
        (0u32..self.order() as u32).flat_map(|u| {
            let u = u as u32;
            self.neighbors(u).map(move |v| Edge(u, v))
        })
    }

    fn from_buffer(buf: &mut [Edge]) -> Self
    {
        // Sort the buffer in lexigraphical order.
        buf.par_sort_unstable();

        let num_nodes = match (
            par::max(buf.par_iter().map(|e| e.0)),
            par::max(buf.par_iter().map(|e| e.1)),
        ) {
            (Some(u), Some(v)) => std::cmp::max(u, v) as usize,
            (None, Some(v)) => v as usize,
            (Some(u), None) => u as usize,
            (None, None) => {
                return Self {
                    vertices: vec![],
                    num_edges: 0,
                    edges: vec![],
                }
            }
        };

        let (num_edges, nodes_and_nnzs, mut edges) =
            // Group the buffer into groups of edges that
            // share a source vertex.
            par::group_by(buf, |e1, e2| e1.0 == e2.0)
                .fold(|| (0, vec![], vec![]),
                      |(mut num_edges, mut nodes_and_nnzs, mut edges), group| {
                    let mut iter = group.iter().dedup();
                    if let Some(&Edge(u, v)) = iter.next() {
                        num_edges += 1usize;
                        encoder::encode(
                            &mut edges,
                            u,
                            std::iter::once(v).chain(iter.map(|Edge(_, w)| {
                                num_edges += 1;
                                *w
                            })),
                        );
                        nodes_and_nnzs.push((u, edges.len()));
                        (num_edges, nodes_and_nnzs, edges)
                    }
                    else {
                        (num_edges, nodes_and_nnzs, edges)
                    }
                })
                .reduce(|| (0, vec![], vec![]),
                        |left, right| {
                    (left.0 + right.0, vec::concat(left.1, right.1), vec::concat(left.2, right.2))
                });

        edges.shrink_to_fit();

        let mut nnzs = Vec::with_capacity(nodes_and_nnzs.len());

        for (u, nnz) in nodes_and_nnzs {
            nnzs.resize(u as usize, 0);
            nnzs.push(nnz);
        }

        nnzs.resize(num_nodes + 1, 0);

        let mut vertices = par::exclusive_sum(0, nnzs);

        vertices.shrink_to_fit();

        Self {
            vertices,
            num_edges,
            edges,
        }
    }

    pub fn nbytes(&self) -> usize
    {
        let mut bytes = std::mem::size_of_val(self);
        bytes += std::mem::size_of_val(&self.vertices[..]);
        bytes += std::mem::size_of_val(&self.edges[..]);
        bytes
    }

    /// The number of vertices in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip::CSR;
    ///
    /// let csr = CSR::from(vec![
    ///     [0u32,1],
    ///     [0,2],
    ///     [1,0],
    ///     [2,1],
    ///     [0,3]
    /// ]);
    ///
    /// assert_eq!(4, csr.order());
    /// ```
    pub fn order(&self) -> usize
    {
        self.vertices.len().saturating_sub(1)
    }

    /// The number of edges in the graph.
    pub fn size(&self) -> usize
    {
        self.num_edges
    }

    pub fn try_from_edges<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Edge, E>>,
    {
        let edges = iter.into_iter();
        let mut buf = Vec::new();
        for edge in edges {
            match edge {
                Ok(e) => buf.push(e),
                Err(err) => return Err(err),
            };
        }
        Ok(Self::from_buffer(&mut buf[..]))
    }

    pub fn try_from_edges_with_capacity<E, I>(capacity: usize, iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Edge, E>>,
    {
        let edges = iter.into_iter();
        let mut buf = Vec::with_capacity(capacity);
        for edge in edges {
            match edge {
                Ok(e) => buf.push(e),
                Err(err) => return Err(err),
            };
        }
        Ok(Self::from_buffer(&mut buf[..]))
    }
}

impl<T: Into<Edge>> From<Vec<T>> for CSR
{
    fn from(vect: Vec<T>) -> Self
    {
        let mut edges: Vec<Edge> = vect.into_iter().map(|x| x.into()).collect();
        Self::from_buffer(&mut edges[..])
    }
}
