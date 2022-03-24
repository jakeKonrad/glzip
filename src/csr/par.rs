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

use std::{
    ops::Add,
    slice,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use crossbeam_utils::CachePadded;
use rayon::prelude::*;

use crate::{encoder, iter::*, vec, Edge, CSR};

fn prefix_sum<T>(vect: Vec<T>) -> Vec<T>
where
    T: Add<Output = T> + Copy + Sized + Send + Default,
{
    rayon::iter::once(T::default())
        .chain(vect.into_par_iter())
        .fold(Vec::new, |mut acc, x| {
            match acc.last() {
                Some(&sum) => acc.push(sum + x),
                None => acc.push(x),
            };
            acc
        })
        .reduce(Vec::new, |mut left, right| match left.last() {
            Some(&sum) => {
                left.extend(right.into_iter().map(|x| x + sum));
                left
            }
            None => right,
        })
}

fn find_index_edgelist(xs: &[Edge]) -> Option<usize>
{
    let n = xs.len() / 2;

    for (start, end) in (0..).scan(true, |cont, i| {
        if *cont {
            let offset = 2usize.pow(i);
            let start = n.saturating_sub(offset);
            let end = n + offset;
            Some(
                if !(1..xs.len()).contains(&start) || !(2..xs.len()).contains(&end) {
                    *cont = false;
                    (0, xs.len())
                }
                else {
                    (start, end)
                },
            )
        }
        else {
            None
        }
    }) {
        match xs[start..end]
            .windows(2)
            .enumerate()
            .find_map(|(i, win)| {
                if win[0].0 == win[1].0 {
                    None
                }
                else {
                    Some(i)
                }
            })
            .map(|i| start + i)
        {
            Some(i) => return Some(i),
            None => {}
        }
    }
    None
}

fn split_edgelist(xs: &[Edge]) -> (&[Edge], Option<&[Edge]>)
{
    match find_index_edgelist(xs) {
        Some(i) => {
            let (ys, zs) = xs.split_at(i + 1);
            (ys, Some(zs))
        }
        None => (xs, None),
    }
}

fn rec_edgelist_to_csr(
    edgelist: &[Edge],
    global_num_edges: &AtomicUsize,
) -> (Vec<(u32, usize)>, Vec<u8>)
{
    match split_edgelist(edgelist) {
        (xs, None) => {
            let mut iter = xs.iter().dedup();
            if let Some(&Edge(u, v)) = iter.next() {
                let mut edges = Vec::with_capacity(xs.len());
                let mut num_edges = 1usize;
                encoder::encode(
                    &mut edges,
                    u,
                    std::iter::once(v).chain(iter.map(|Edge(_, w)| {
                        num_edges += 1;
                        *w
                    })),
                );
                global_num_edges.fetch_add(num_edges, Ordering::Relaxed);
                (vec![(u, edges.len())], edges)
            }
            else {
                (vec![], vec![])
            }
        }
        (xs, Some(ys)) => {
            let (l, r) = rayon::join(
                || rec_edgelist_to_csr(xs, global_num_edges),
                || rec_edgelist_to_csr(ys, global_num_edges),
            );
            (vec::concat(l.0, r.0), vec::concat(l.1, r.1))
        }
    }
}

pub fn edgelist_to_csr(edgelist: &mut [Edge]) -> (Vec<usize>, usize, Vec<u8>)
{
    edgelist.par_sort_unstable();

    let global_num_edges = AtomicUsize::new(0usize);

    let (nodes_and_nnzs, mut edges) = rec_edgelist_to_csr(edgelist, &global_num_edges);

    edges.shrink_to_fit();

    let mut nnzs = Vec::with_capacity(nodes_and_nnzs.len());

    for (u, nnz) in nodes_and_nnzs {
        nnzs.resize(u as usize, 0);
        nnzs.push(nnz);
    }

    let mut indptr = prefix_sum(nnzs);

    indptr.shrink_to_fit();

    (indptr, global_num_edges.into_inner(), edges)
}

fn calc_prop(v: u32, mut sizes: slice::Iter<'_, usize>, graph: &CSR, p: &[CachePadded<AtomicU64>])
{
    if let Some(&k) = sizes.next() {
        for u in graph.adj(v) {
            rayon::join(
                || calc_prop(u, sizes.clone(), graph, p),
                || {
                    let prob = k as f64 / (std::cmp::max(graph.degree(v), k) as f64);
                    let x = &p[u as usize];
                    let mut prev_prob = x.load(Ordering::Relaxed);
                    loop {
                        let new_prob = (f64::from_bits(prev_prob) + prob).to_bits();
                        match x.compare_exchange_weak(
                            prev_prob,
                            new_prob,
                            Ordering::SeqCst,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(prev) => prev_prob = prev,
                        }
                    }
            });
        }
    }
}


pub fn probability_calculation(graph: &CSR, train_idx: &[bool], sizes: &[usize]) -> Vec<f64>
{
    let mut p = Vec::with_capacity(train_idx.len());

    for &b in train_idx {
        if b {
            p.push(CachePadded::new(AtomicU64::new(1f64.to_bits())));
        }
        else {
            p.push(CachePadded::new(AtomicU64::new(0f64.to_bits())));
        }
    }

    let sizes = sizes.iter();

    (0..train_idx.len()).into_par_iter().for_each(|v| {
        if train_idx[v] {
            calc_prop(v as u32, sizes.clone(), graph, &p[..]);
        }
    });

    p.into_iter()
        .map(|shared_float| f64::from_bits(shared_float.into_inner().into_inner()))
        .collect()
}
