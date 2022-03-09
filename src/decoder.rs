// glzip is a graph compression library for graph learning systems.
// Copyright (C) 2022  Jacob Konrad
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

// Based on:
//
// Smaller and Faster: Parallel Processing of
// Compressed Graphs with Ligra+
//
// by Julian Shun, Laxman Dhulipala and Guy E. Blelloch
//
// http://www.cs.cmu.edu/~jshun/ligra+.pdf

// The main idea is to compress the graphs like they do
// in Ligra+ so that large graphs can fit into memory and
// therefore achieve a speed up.
//

use std::{iter, slice::Iter};

fn first_edge(source: u32, bytes: &mut Iter<'_, u8>) -> Option<u32>
{
    match bytes.next() {
        None => None,
        Some(&first_byte) => {
            let mut diff = (first_byte & 0x3f) as u32;
            if (first_byte & 0x80) > 0 {
                let mut shift_amount = 6;
                loop {
                    match bytes.next() {
                        Some(&b) => {
                            diff |= ((b & 0x7f) as u32) << shift_amount;
                            if (b & 0x80) > 0 {
                                shift_amount += 7;
                            }
                            else {
                                break;
                            }
                        }
                        None => unreachable!(),
                    }
                }
            }
            Some(if first_byte & 0x40 > 0 {
                source - diff
            }
            else {
                source + diff
            })
        }
    }
}

fn next_group(prev_edge: &mut u32, bytes: &mut Iter<'_, u8>) -> impl Iterator<Item = u32>
{
    (match bytes.next() {
        None => None,
        Some(&header) => {
            let num_bytes = ((header & 0x3) + 1) as usize;
            let run_length = ((header >> 2) + 1) as usize;
            let (left, right) = bytes.as_slice().split_at(num_bytes * run_length);
            *bytes = right.iter();
            let mut buf = [0u32; 64];
            match num_bytes {
                1 => {
                    for (i, chunk) in left.chunks_exact(1).enumerate() {
                        buf[i] = chunk[0] as u32;
                    }
                }
                2 => {
                    for (i, chunk) in left.chunks_exact(2).enumerate() {
                        let mut diff = (chunk[0] as u32) << 8;
                        diff |= chunk[1] as u32;
                        buf[i] = diff;
                    }
                }
                3 => {
                    for (i, chunk) in left.chunks_exact(3).enumerate() {
                        let mut diff = (chunk[0] as u32) << 16;
                        diff |= (chunk[1] as u32) << 8;
                        diff |= chunk[2] as u32;
                        buf[i] = diff;
                    }
                }
                4 => {
                    for (i, chunk) in left.chunks_exact(4).enumerate() {
                        let mut diff = (chunk[0] as u32) << 24;
                        diff |= (chunk[1] as u32) << 16;
                        diff |= (chunk[2] as u32) << 8;
                        diff |= chunk[3] as u32;
                        buf[i] = diff;
                    }
                }
                _ => unreachable!(),
            }

            let mut p_edge = *prev_edge;
            for i in 0..run_length {
                let sum = p_edge + buf[i];
                p_edge = sum;
                buf[i] = sum;
            }
            *prev_edge = p_edge;

            Some((0..run_length).map(move |i| buf[i]))
        }
    })
    .into_iter()
    .flatten()
}

pub fn next_edges(mut prev_edge: u32, bytes: &[u8]) -> impl Iterator<Item = u32> + '_
{
    let mut bytes_iter = bytes.iter();
    iter::from_fn(move || {
        if bytes_iter.as_slice().is_empty() {
            None
        }
        else {
            Some(next_group(&mut prev_edge, &mut bytes_iter))
        }
    })
    .flatten()
}

pub fn decode(source: u32, bytes: &[u8]) -> impl Iterator<Item = u32> + '_
{
    let mut iter = bytes.iter();
    let edge = first_edge(source, &mut iter);
    let n_edges = edge.map(|e| next_edges(e, iter.as_slice()));
    edge.into_iter().chain(n_edges.into_iter().flatten())
}
