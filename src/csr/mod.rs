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

use std::convert::TryInto;

use crate::{
    decoder,
    error::{Error, Void},
    iter::IteratorFlipExt,
    Edge,
};

mod par;

pub struct Adj<'a>(Option<decoder::Decoder<'a>>);

impl<'a> Iterator for Adj<'a>
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item>
    {
        match &mut self.0 {
            Some(iter) => iter.next(),
            None => None,
        }
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
    /// The edges adjacent to a vertex.
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
    /// assert_eq!(vec![1,2], csr.adj(0).collect::<Vec<_>>());
    /// assert_eq!(vec![0], csr.adj(1).collect::<Vec<_>>());
    /// assert_eq!(vec![1], csr.adj(2).collect::<Vec<_>>());
    /// ```
    pub fn adj(&self, source: u32) -> Adj<'_>
    {
        let i = source as usize;
        Adj(self.vertices.get(i).and_then(move |&start| {
            self.vertices
                .get(i + 1)
                .map(|&end| decoder::decode(source, &self.edges[start..end]))
        }))
    }

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

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_
    {
        (0u32..self.order() as u32).flat_map(|u| {
            let u = u as u32;
            self.adj(u).map(move |v| Edge(u, v))
        })
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

    pub fn nbytes(&self) -> usize
    {
        let mut bytes = std::mem::size_of_val(self);
        bytes += std::mem::size_of_val(&self.vertices[..]);
        bytes += std::mem::size_of_val(&self.edges[..]);
        bytes
    }

    /// The number of vertices in the graph.
    pub fn order(&self) -> usize
    {
        self.vertices.len().saturating_sub(1)
    }

    pub fn optimize(&self, train_idx: &[bool], sizes: &[usize]) -> (Self, Vec<u32>)
    {
        let probs = par::probability_calculation(self, train_idx, sizes);

        println!("{:?}", probs);

        let mut vs: Vec<u32> = (0u32..self.order() as u32).collect();

        // sort by descending probability
        vs.sort_unstable_by(|&a, &b| probs[b as usize].total_cmp(&probs[a as usize]));

        (self.reorder(&vs[..]), vs)
    }

    fn reverse(&self) -> Self
    {
        let iter = self.edges().map(|e| e.into()).flip().map(|t| Ok(t.into()));

        let graph: Result<Self, !> = Self::try_from_edges_with_capacity(self.size(), iter);

        graph.into_ok()
    }

    fn reorder(&self, perm: &[u32]) -> Self
    {
        let iter = self
            .edges()
            .map(|e| Ok(Edge(perm[e.0 as usize], perm[e.1 as usize])));

        let graph: Result<Self, !> = Self::try_from_edges_with_capacity(self.size(), iter);

        graph.into_ok()
    }

    /// The number of edges in the graph.
    pub fn size(&self) -> usize
    {
        self.num_edges
    }

    pub fn try_from_csr<Ix, T>(indptr: &[Ix], indices: &[T]) -> Result<Self, Error<Ix, T>>
    where
        Ix: TryInto<usize> + Copy,
        T: TryInto<u32> + Copy,
    {
        let mut buf = Vec::with_capacity(indices.len());
        let range = indptr.len().saturating_sub(1);
        if range <= u32::MAX as usize {
            for i in 0..range {
                let start = match indptr[i].try_into() {
                    Ok(x) => x,
                    Err(e) => return Err(Error::TryIntoUsize(e)),
                };

                let end = match indptr[i + 1].try_into() {
                    Ok(x) => x,
                    Err(e) => return Err(Error::TryIntoUsize(e)),
                };

                for edge in indices[start..end].iter().map(|&v| v.try_into()) {
                    match edge {
                        Ok(v) => buf.push(Edge(i as u32, v)),
                        Err(e) => return Err(Error::TryIntoU32(e)),
                    };
                }
            }
            Ok(Self::from_buffer(&mut buf[..]))
        }
        else {
            Err(Error::TooManyVertices)
        }
    }

    pub fn try_from_edge_index<T>(src: &[T], dst: &[T]) -> Result<Self, Error<Void, T>>
    where
        T: TryInto<u32> + Copy,
    {
        if src.len() <= u32::MAX as usize && dst.len() <= u32::MAX as usize {
            let iter = src.iter().zip(dst.iter()).map(|a| {
                (*a.0)
                    .try_into()
                    .and_then(|u| (*a.1).try_into().map(|v| Edge(u, v)))
                    .map_err(Error::TryIntoU32)
            });

            let capacity = std::cmp::min(src.len(), dst.len());

            Self::try_from_edges_with_capacity(capacity, iter)
        }
        else {
            Err(Error::TooManyVertices)
        }
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
