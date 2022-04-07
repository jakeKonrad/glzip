
use std::{slice, sync::atomic::{AtomicU64, Ordering}};

use rayon::prelude::*;

use crate::{CSR, Edge};

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
) {
    if let Some(&k) = sizes.next() {
        let v_ix = v as usize; 
        if in_degree[v_ix] < threshold {
            let prob = (k as f64 / (std::cmp::max(in_degree[v_ix], k) as f64)) * weight;
            for u in incoming.neighbors(v) {
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

pub fn by_access_probabilites(csr: &CSR, train_idx: &[bool], sizes: &[usize]) -> (CSR, Vec<u32>)
{
    let incoming = CSR::from_edges_with_capacity(csr.size(), csr.edges().map(|e| Edge(e.1, e.0)));

    let threshold = (csr.order() as f64).sqrt().ceil() as usize;

    let num_nodes = csr.order() as u32;

    let out_degree: Vec<usize> = (0u32..num_nodes)
        .into_par_iter()
        .map(|v| csr.degree(v))
        .collect();

    let in_degree: Vec<usize> = (0u32..num_nodes)
        .into_par_iter()
        .map(|v| incoming.degree(v))
        .collect();

    let p: Vec<AtomicU64> =
        std::iter::repeat_with(|| AtomicU64::new(0f64.to_bits()))
            .take(csr.order())
            .collect();

    let sizes = sizes.iter();

    (0..csr.order()).into_par_iter().for_each(|v| {
        if train_idx[v] {
            calc_prob(v as u32, 1f64, sizes.clone(), &incoming, threshold, &in_degree[..], &out_degree[..], &p[..]);
        }
    });

    let probs: Vec<f64> = p.into_iter()
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
        .collect();

    let mut vs: Vec<u32> = (0u32..csr.order() as u32).collect();

    vs.par_sort_unstable_by(|&a, &b| probs[b as usize].total_cmp(&probs[a as usize]));

    let new_csr = CSR::from_edges_with_capacity(csr.size(), csr.edges().map(|e| Edge(vs[e.0 as usize], vs[e.1 as usize])));

    (new_csr, vs)
}
