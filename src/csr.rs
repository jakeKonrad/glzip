//! Compress Sparse Row representation of directed graphs but even more compressed as the 
//! edges are compressed using the byte run length encoding scheme from 
//! [Ligra+](https://people.csail.mit.edu/jshun/ligra+.pdf).

use std::{
    ops::BitOr,
    cmp,
    collections::BTreeSet,
};

use rayon::prelude::*;

use crate::{encoder, decoder, Edge};

fn take_vec<T, E, I: Iterator<Item = Result<T, E>>>(xs: &mut I, n: usize) -> Result<Vec<T>, E>
{
    let mut ys = vec![];
    for _ in 0..n {
        match xs.next() {
            Some(Ok(a)) => ys.push(a),
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    Ok(ys)
}

/// The Compressed Sparse Row struct.
pub struct CSR
{
    vertices: Vec<usize>,
    edges: Vec<u8>,
}

impl CSR
{
    /// Create an empty CSR mostly useful as a unit for the `bitor` function. 
    pub fn new() -> Self
    {
        Self {
            vertices: vec![],
            edges: vec![],
        }
    }

    /// The neighbors of a vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use glzip_core::csr::CSR;
    ///
    /// let edges = vec![
    ///     [0,1],
    ///     [0,2],
    ///     [1,0],
    ///     [2,1],
    /// ];
    /// 
    /// let csr = CSR::from(edges);
    ///
    /// assert_eq!(vec![1,2], csr.neighbors(0).collect::<Vec<_>>());
    pub fn neighbors(&self, source: u32) -> impl Iterator<Item = u32> + '_ 
    {
        let i = source as usize;
        self.vertices.get(i)
            .into_iter()
            .flat_map(move |&start| {
                self.vertices.get(i + 1)
                    .into_iter()
                    .flat_map(move |&end| decoder::decode(source, &self.edges[start..end]))
            })
    }

    /// Since this is a compression algorithm you might want to try
    /// and use it from graphs that do not fit in memory uncompressed.
    /// This function lets you compress a graph in chunks.
    pub fn out_of_core_from_edges<I, E>(
        edges_per_chunk: usize,
        source: I,
    ) -> Result<Self, E>
    where I: Iterator<Item = Result<Edge, E>>
    {
         let mut edges = source.peekable();
         let mut csr = Self::new();
         while edges.peek().is_some() {
             let chunk = match take_vec(&mut edges, edges_per_chunk) {
                 Ok(c) => c,
                 Err(e) => return Err(e),
             };
             csr = &csr | &Self::from(chunk)
         }
         Ok(csr)
    }

    pub fn nbytes(&self) -> usize 
    {
        let mut bytes = std::mem::size_of_val(self);
        bytes += std::mem::size_of_val(&self.vertices[..]);
        bytes += std::mem::size_of_val(&self.edges[..]);
        bytes
    }
}

impl From<Vec<Edge>> for CSR
{
    /// Create a CSR from a `Vec` of edges.
    fn from(mut edges: Vec<Edge>) -> Self
    {
        edges.par_sort_unstable();
        edges
            .into_par_iter()
            // Create an adjacency list approximately per core/thread
            .fold(|| vec![], |mut adj_list: Vec<Vec<u32>>, [u, v]| {
                let u = u as usize;
                adj_list.resize_with(cmp::max(adj_list.len(), u + 1), Vec::new);
                // Filter out duplicate edges
                if adj_list[u].last().map(|&w| w != v).unwrap_or(true) {
                    adj_list[u].push(v);
                }
                adj_list
            })
            // Turn each adjacency list into a CSR
            .map(|adj_list| {
                let mut vertices = Vec::with_capacity(adj_list.len() + 1);
                let mut edges = vec![];

                for (u, vs) in adj_list.into_iter().enumerate() {
                    vertices.push(edges.len());
                    encoder::encode(&mut edges, u as u32, vs);
                }

                vertices.push(edges.len());

                vertices.shrink_to_fit();
                edges.shrink_to_fit();

                Self {
                    vertices,
                    edges,
                }
            })
            .reduce(|| Self::new(), |a, b| &a | &b)
    }
}

fn unioning<T, I, J>(left: I, right: J) -> Vec<T>
where T: Ord + Copy,
      I: IntoIterator<Item = T>,
      J: IntoIterator<Item = T>
{
    let xs: BTreeSet<T> = left.into_iter().collect();
    let ys: BTreeSet<T> = right.into_iter().collect();
    xs.union(&ys).copied().collect()
}


impl BitOr for &CSR 
{
    type Output = CSR;

    /// The union of two graphs.
    fn bitor(self, rhs: Self) -> CSR
    {
        let new_len = cmp::max(self.vertices.len(), rhs.vertices.len());
        let (nnz, mut edges): (Vec<usize>, Vec<u8>) = 
            (0..new_len)
            .into_par_iter()
            .fold(|| (vec![], vec![]), |(mut nnz, mut edges), u| {
                let u = u as u32;
                let pre = edges.len();
                encoder::encode(&mut edges, u, unioning(self.neighbors(u), rhs.neighbors(u)));
                let post = edges.len();
                nnz.push(post - pre);
                (nnz, edges)
            })
            .reduce(|| (vec![], vec![]), |(mut nnz, mut edges), b| {
                nnz.extend_from_slice(&b.0[..]);
                edges.extend_from_slice(&b.1[..]);
                (nnz, edges)
            });

        let mut vertices: Vec<usize> = Vec::with_capacity(nnz.len() + 1);
        
        vertices.push(0);
        vertices.extend(
            nnz
            .into_iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
        );

        edges.shrink_to_fit();

        CSR {
            vertices,
            edges,
        }
    }
}

