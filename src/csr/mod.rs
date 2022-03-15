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

//! Compress Sparse Row representation of directed graphs but even more compressed as the
//! edges are compressed using the byte run length encoding scheme from
//! [Ligra+](https://people.csail.mit.edu/jshun/ligra+.pdf).

use std::{
    cmp,
    error::Error,
    ops::{BitOr, BitOrAssign},
};

use crate::{decoder, encoder, iter::*, Edge, BYTES_PER_EDGE, DEFAULT_EDGES_PER_CHUNK};

//mod seq;

mod par;

pub struct CSRBuilder
{
    num_threads: usize,
    edges_per_chunk: usize,
}

impl CSRBuilder
{
    pub fn new() -> Self
    {
        Self {
            num_threads: num_cpus::get(),
            edges_per_chunk: DEFAULT_EDGES_PER_CHUNK,
        }
    }

    pub fn num_threads(self, num_threads: usize) -> Self
    {
        Self {
            num_threads,
            edges_per_chunk: self.edges_per_chunk,
        }
    }

    pub fn edges_per_chunk(self, edges_per_chunk: usize) -> Self
    {
        Self {
            num_threads: self.num_threads,
            edges_per_chunk,
        }
    }

    pub fn bytes_per_chunk(self, bytes_per_chunk: usize) -> Result<Self, Self>
    {
        if bytes_per_chunk % BYTES_PER_EDGE == 0 {
            Ok(Self {
                num_threads: self.num_threads,
                edges_per_chunk: bytes_per_chunk / DEFAULT_EDGES_PER_CHUNK,
            })
        }
        else {
            Err(self)
        }
    }

    pub fn build<E, I>(self, iter: I) -> Result<CSR, Box<dyn Error + Send + Sync>>
    where
        E: Into<Box<dyn Error + Send + Sync>>,
        I: IntoIterator<Item = Result<Edge, E>>,
    {
        fn yield_n<T, E, I: Iterator<Item = Result<T, E>>>(
            xs: &mut I,
            n: usize,
            buf: &mut Vec<T>,
        ) -> Result<(), E>
        {
            for _ in 0..n {
                match xs.next() {
                    Some(Ok(x)) => buf.push(x),
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                }
            }
            Ok(())
        }

        let mut edges = iter.into_iter().peekable();
        let mut csr = CSR::new();
        let mut buf = Vec::with_capacity(self.edges_per_chunk);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()?;

        while edges.peek().is_some() {
            match yield_n(&mut edges, self.edges_per_chunk, &mut buf) {
                Ok(()) => {}
                Err(e) => return Err(e.into()),
            };
            csr |= pool.install(|| CSR::from_buffer(&mut buf[..]));
            buf.clear();
        }
        Ok(csr)
    }
}

/// The Compressed Sparse Row struct.
pub struct CSR
{
    vertices: Vec<usize>,
    num_edges: usize,
    edges: Vec<u8>,
}

impl CSR
{
    /// Create an empty CSR mostly useful as a unit for the `bitor` function.
    pub fn new() -> Self
    {
        Self {
            vertices: vec![],
            num_edges: 0,
            edges: vec![],
        }
    }

    /// The number of vertices in the graph.
    pub fn order(&self) -> usize
    {
        self.vertices.len().saturating_sub(1)
    }

    /// The number of edges in the graph.
    pub fn size(&self) -> usize
    {
        self.num_edges
    }

    fn raw_adj(&self, source: u32) -> Option<&[u8]>
    {
        let i = source as usize;
        self.vertices.get(i).and_then(move |&start| {
            self.vertices.get(i + 1).and_then(move |&end| {
                let slice = &self.edges[start..end];
                if slice.len() > 0 {
                    Some(slice)
                }
                else {
                    None
                }
            })
        })
    }

    /// The edges adjacent to a vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip_core::csr::CSR;
    ///
    /// let csr = CSR::from(vec![
    ///     [0,1],
    ///     [0,2],
    ///     [1,0],
    ///     [2,1],
    /// ]);
    ///
    /// assert_eq!(vec![1,2], csr.adj(0).collect::<Vec<_>>());
    /// assert_eq!(vec![0], csr.adj(1).collect::<Vec<_>>());
    /// assert_eq!(vec![1], csr.adj(2).collect::<Vec<_>>());
    /// ```
    pub fn adj(&self, source: u32) -> impl Iterator<Item = u32> + '_
    {
        self.raw_adj(source)
            .into_iter()
            .flat_map(move |slice| decoder::decode(source, slice))
    }

    pub fn nbytes(&self) -> usize
    {
        let mut bytes = std::mem::size_of_val(self);
        bytes += std::mem::size_of_val(&self.vertices[..]);
        bytes += std::mem::size_of_val(&self.edges[..]);
        bytes
    }

    fn from_buffer(buf: &mut [Edge]) -> Self
    {
        let (vertices, num_edges, edges) = par::edgelist_to_csr(buf);

        Self {
            vertices,
            num_edges,
            edges,
        }
    }
}

impl From<Vec<Edge>> for CSR
{
    fn from(mut edges: Vec<Edge>) -> Self
    {
        Self::from_buffer(&mut edges[..])
    }
}

impl BitOr for &CSR
{
    type Output = CSR;

    /// The union of two graphs.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip_core::csr::CSR;
    ///
    /// let g = CSR::from(vec![
    ///     [0,1],
    ///     [0,2],
    ///     [1,2],
    ///     [2,0],
    /// ]);
    ///
    /// let h = CSR::from(vec![
    ///     [0,3],
    ///     [1,3],
    ///     [3,2],
    /// ]);
    ///
    /// let k = g | h;
    ///
    /// assert_eq!(vec![1,2,3], k.adj(0).collect::<Vec<_>>());
    /// assert_eq!(vec![2,3], k.adj(1).collect::<Vec<_>>());
    /// assert_eq!(vec![0], k.adj(2).collect::<Vec<_>>());
    /// assert_eq!(vec![2], k.adj(3).collect::<Vec<_>>());
    /// ```
    fn bitor(self, rhs: Self) -> CSR
    {
        let new_len = cmp::max(self.vertices.len(), rhs.vertices.len());
        let mut vertices = Vec::with_capacity(new_len);
        let mut num_edges = 0usize;
        let mut edges = Vec::new();

        for u in 0..(new_len.saturating_sub(1)) {
            let u = u as u32;
            vertices.push(edges.len());
            match self.raw_adj(u) {
                Some(vs) => match rhs.raw_adj(u) {
                    Some(ws) => {
                        let vs = decoder::decode(u, vs);
                        let ws = decoder::decode(u, ws);
                        encoder::encode(
                            &mut edges,
                            u,
                            vs.union(ws).map(|v| {
                                num_edges += 1;
                                v
                            }),
                        );
                    }
                    None => {
                        edges.extend(vs.iter().copied());
                    }
                },
                None => match rhs.raw_adj(u) {
                    Some(vs) => {
                        edges.extend(vs.iter().copied());
                    }
                    None => {}
                },
            };
        }

        vertices.push(edges.len());

        vertices.shrink_to_fit();
        edges.shrink_to_fit();

        CSR {
            vertices,
            num_edges,
            edges,
        }
    }
}

impl BitOr for CSR
{
    type Output = CSR;

    fn bitor(self, rhs: Self) -> CSR
    {
        if self.vertices.len() == 0 {
            return rhs;
        }

        if rhs.vertices.len() == 0 {
            return self;
        }

        &self | &rhs
    }
}

impl BitOrAssign for CSR
{
    fn bitor_assign(&mut self, rhs: Self)
    {
        if self.vertices.len() == 0 {
            *self = rhs;
            return;
        }

        if rhs.vertices.len() == 0 {
            return;
        }

        *self = self as &CSR | &rhs
    }
}
