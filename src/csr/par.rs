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
    collections::{HashSet, BinaryHeap},
};

use rayon::prelude::*;

use crate::{encoder, iter::*, vec, Edge, CSR, slice::*};

pub fn exclusive_sum<T>(init: T, vect: Vec<T>) -> Vec<T>
where
    T: Add<Output = T> + Copy + Sized + Send,
{
    rayon::iter::once(init)
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
            let offset = 2 * i;
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

    let mut indptr = exclusive_sum(0, nnzs);

    indptr.shrink_to_fit();

    (indptr, global_num_edges.into_inner(), edges)
}

#[inline]
fn atomic_add_f64(x: &AtomicU64, y: f64)
{
    let mut old = x.load(Ordering::Relaxed);
    loop {
        let new = (y + f64::from_bits(old)).to_bits();
        match x.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(prev) => old = prev,
        }
    }
}

fn calc_prob(
    v: u32,
    weight: f64,
    mut sizes: slice::Iter<'_, usize>,
    incoming: &CSR,
    threshold: usize,
    in_degree: &[usize],
    out_degree: &[usize],
    p: &[AtomicU64]
)
{
    if let Some(&k) = sizes.next() {
        let v_ix = v as usize;
        let v_in_degree = in_degree[v_ix];
        if v_in_degree < threshold {
            let prob = (k as f64 / (std::cmp::max(v_in_degree, k) as f64)) * weight;
            for u in incoming.adj(v) {
                let u_ix = u as usize;
                if out_degree[u_ix] < threshold {
                    atomic_add_f64(&p[u_ix], prob);
                    calc_prob(
                        u,
                        prob,
                        sizes.clone(),
                        incoming,
                        threshold,
                        in_degree,
                        out_degree,
                        p
                    );
                }
            }
        }
    }
}

pub fn probability_calculation(graph: &CSR, train_idx: &[bool], sizes: &[usize]) -> Vec<f64>
{
    let incoming = graph.reverse();
    let threshold = (graph.order() as f64).sqrt().ceil() as usize;

    let num_nodes = graph.order() as u32;

    let out_degree: Vec<usize> = (0u32..num_nodes)
        .into_par_iter()
        .map(|v| graph.degree(v))
        .collect();

    let in_degree: Vec<usize> = (0u32..num_nodes)
        .into_par_iter()
        .map(|v| incoming.degree(v))
        .collect();

    let p: Vec<AtomicU64> =
        std::iter::repeat_with(|| AtomicU64::new(0f64.to_bits()))
            .take(graph.order())
            .collect();

    let sizes = sizes.iter();

    (0..graph.order()).into_par_iter().for_each(|v| {
        if train_idx[v] {
            calc_prob(
                v as u32,
                1f64,
                sizes.clone(),
                &incoming,
                threshold,
                &in_degree[..],
                &out_degree[..],
                &p[..]
            );
        }
    });

    p.into_par_iter()
        .enumerate()
        .map(|(v, shared_float)| {
            if out_degree[v] >= threshold {
                f64::MAX
            }
            else {
                let prob = (train_idx[v] as u64) as f64;
                prob + f64::from_bits(shared_float.into_inner())
            }
        })
        .collect()
}

pub fn reordering(graph: &CSR, k: usize, num_nodes: u32) -> Vec<u32>
{
    let vs: Vec<u32> = (0u32..num_nodes).collect();
    let mut new_vs = Vec::with_capacity(vs.len());

    struct Prio
    {
        key: usize,
        val: u32,
    }

    impl PartialEq for Prio {

        fn eq(&self, other: &Self) -> bool
        {
            self.key == other.key
        }
    }

    impl Eq for Prio { }

    impl Ord for Prio
    {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering
        {
            self.key.cmp(&other.key)
        }
    }

    impl PartialOrd for Prio
    {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering>
        {
            Some(self.cmp(other))
        }
    }

    for slice in vs.exponential_chunks(k) {
        let set: HashSet<u32> = slice.iter().copied().collect();

        let mut scores: BinaryHeap<Prio> = slice.into_par_iter()
            .map(|&v| {
                let score = graph.adj(v).filter(|u| set.contains(u)).count();
                Prio { key: score, val: v }
            })
            .collect();

        while let Some(prio) = scores.pop() {
            new_vs.push(prio.val);
        }
    }

    new_vs
}

