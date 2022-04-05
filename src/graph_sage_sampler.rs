
use std::{slice, collections::HashMap};

use crate::{CSR, iter::*, vec};

use rayon::prelude::*;

pub struct Adj
{
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
    pub size: (usize, usize),
}

pub struct GraphSageSampler<'a>
{
    csr: &'a CSR,
    sizes: slice::Iter<'a, usize>,
}

impl<'a> GraphSageSampler<'a> 
{
    pub fn new(csr: &'a CSR, sizes: &'a [usize]) -> Self
    {
        Self { csr, sizes: sizes.iter() }
    }

    fn sample_kernel(&self, inputs: &[u32], k: usize) -> (Vec<u32>, Vec<usize>)
    {
        inputs
            .par_iter()
            .map_init(|| rand::thread_rng(), |rng, &v| {
                let ns = self.csr.neighbors(v).reservoir_sample(rng, k);
                let d = ns.len();
                (ns, vec![d])
            })
            .reduce(|| (vec![], vec![]), |a, b| (vec::concat(a.0, b.0), vec::concat(a.1, b.1)))
    }

    fn reindex(&self, inputs: &[u32], outputs: &[u32], output_counts: &[usize]) -> (Vec<u32>, Vec<u32>, Vec<u32>)
    {
        let mut out_map: HashMap<u32, u32> = HashMap::new();

        let mut frontier = Vec::new();

        let mut n_id = 0;

        for &input in inputs.iter() {
            out_map.insert(input, n_id);
            n_id += 1;
            frontier.push(input);
        }

        for &output in outputs.iter() {
            if !out_map.contains_key(&output) {
                out_map.insert(output, n_id); 
                n_id += 1;
                frontier.push(output);
            }
        }

        let mut row_idx = Vec::with_capacity(outputs.len());
        let mut col_idx = Vec::with_capacity(outputs.len());

        let mut cnt = 0;
        for &input in inputs.iter() {
            for _ in 0..output_counts[input as usize] {
                row_idx.push(out_map[&input]);
                col_idx.push(out_map[&outputs[cnt]]);
                cnt += 1;
            }
        }
        (frontier, row_idx, col_idx)
    }   

    pub fn sample(&self, input_nodes: &[u32]) -> (Vec<u32>, usize, Vec<Adj>)
    {
        let mut nodes: Vec<_> = input_nodes.iter().copied().collect();
        let mut adjs = Vec::new();
        let batch_size = nodes.len();

        for &k in self.sizes.clone() {
            let (out, cnt) = self.sample_kernel(&nodes[..], k);
            let (frontier, src, dst) = self.reindex(&nodes[..], &out[..], &cnt[..]);
            let size = (frontier.len(), nodes.len());
            adjs.push(Adj { src, dst, size });
            nodes = frontier;
        }

        adjs.reverse();

        (nodes, batch_size, adjs)
    }
}

