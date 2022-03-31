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

use std::{
    mem::MaybeUninit,
    ops::Range,
    slice::Iter,
};

use rayon::iter::IntoParallelIterator;
 use rayon::iter::ParallelIterator;

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

fn next_groups_map<'a, OP>(
    prev_edge: u32,
    mut bytes: Iter<'a, u8>, 
    mut op: OP
)
where
    OP: FnMut(u32),
{
    match bytes.next() {
        None => {},
        Some(&header) => {
            let num_bytes = ((header & 0x3) + 1) as usize;
            let run_length = ((header >> 2) + 1) as usize;
            let (left, right) = bytes.as_slice().split_at(num_bytes * run_length);

            let mut p_edge = prev_edge;
            match num_bytes {
                1 => {
                    for &byte in left.iter() {
                        let diff = byte as u32;
                        let sum = p_edge + diff;
                        p_edge = sum;
                        op(sum);
                    }
                }
                2 => {
                    for chunk in left.chunks_exact(2) {
                        let mut diff = (chunk[0] as u32) << 8;
                        diff |= chunk[1] as u32;
                        let sum = p_edge + diff;
                        p_edge = sum;
                        op(sum);
                    }
                }
                3 => {
                    for chunk in left.chunks_exact(3) {
                        let mut diff = (chunk[0] as u32) << 16;
                        diff |= (chunk[1] as u32) << 8;
                        diff |= chunk[2] as u32;
                        let sum = p_edge + diff;
                        p_edge = sum;
                        op(sum);
                    }
                }
                4 => {
                    for chunk in left.chunks_exact(4) {
                        let mut diff = (chunk[0] as u32) << 24;
                        diff |= (chunk[1] as u32) << 16;
                        diff |= (chunk[2] as u32) << 8;
                        diff |= chunk[3] as u32;
                        let sum = p_edge + diff;
                        p_edge = sum;
                        op(sum);
                    }
                }
                _ => unreachable!(),
            }

            next_groups_map(p_edge, right.iter(), op);
        }
    }
}

pub fn decode_map<OP>(source: u32, bytes: &[u8], mut op: OP)
where
    OP: FnMut(u32),
{
    let mut bytes = bytes.iter();
    let mut prev_edge = match first_edge(source, &mut bytes) {
        Some(e) => {
            op(e);
            e
        }
        None => return,
    };

    next_groups_map(prev_edge, bytes, op);
}

fn group_map_par<'a, OP>(
    buf: [MaybeUninit<u32>; 64],
    run_length: usize,
    op: OP)
where
    OP: Fn(u32),
{
    buf[0..run_length]
        .for_each(|u| {
            let u = unsafe { u.assume_init() };
            op(u);
        });
}

fn next_groups_map_par<'a, OP>(
    prev_edge: u32,
    bytes: &[u8], 
    op: OP
)
where
    OP: Fn(u32) + Sync + Send + Copy,
{
    let mut bytes = bytes.iter();
    match bytes.next() {
        None => {},
        Some(&header) => {
            let num_bytes = ((header & 0x3) + 1) as usize;
            let run_length = ((header >> 2) + 1) as usize;
            let (left, right) = bytes.as_slice().split_at(num_bytes * run_length);

            let mut buf: [MaybeUninit<u32>; 64] = unsafe { MaybeUninit::uninit().assume_init() };

            match num_bytes {
                1 => {
                    for (&byte, dst) in left.iter().zip(buf.iter_mut()) {
                        dst.write(byte as u32);
                    }
                }
                2 => {
                    for (chunk, dst) in left.chunks_exact(2).zip(buf.iter_mut()) {
                        let mut diff = (chunk[0] as u32) << 8;
                        diff |= chunk[1] as u32;
                        dst.write(diff);
                    }
                }
                3 => {
                    for (chunk, dst) in left.chunks_exact(3).zip(buf.iter_mut()) {
                        let mut diff = (chunk[0] as u32) << 16;
                        diff |= (chunk[1] as u32) << 8;
                        diff |= chunk[2] as u32;
                        dst.write(diff);
                    }
                }
                4 => {
                    for (chunk, dst) in left.chunks_exact(4).zip(buf.iter_mut()) {
                        let mut diff = (chunk[0] as u32) << 24;
                        diff |= (chunk[1] as u32) << 16;
                        diff |= (chunk[2] as u32) << 8;
                        diff |= chunk[3] as u32;
                        dst.write(diff);
                    }
                }
                _ => unreachable!(),
            }

            let mut p_edge = prev_edge;
            for item in buf.iter_mut().take(run_length) {
                let diff = unsafe { item.assume_init() };
                let edge = p_edge + diff;
                p_edge = edge;
                item.write(edge);
            }

            rayon::join(|| group_map_par(buf, run_length, op),
                        || next_groups_map_par(p_edge, right, op));
        }
    }
}

pub fn decode_map_par<OP>(source: u32, bytes: &[u8], op: OP)
where
    OP: Fn(u32) + Sync + Send + Copy,
{
    let mut bytes = bytes.iter();
    let prev_edge = match first_edge(source, &mut bytes) {
        Some(e) => e,
        None => return,
    };

    rayon::join(|| op(prev_edge),
                || next_groups_map_par(prev_edge, bytes.as_slice(), op));
}

pub fn count(source: u32, bytes: &[u8]) -> usize
{
    let mut bytes = bytes.iter();

    match first_edge(source, &mut bytes) {
        None => return 0,
        Some(_) => {},
    }

    let mut acc = 1;

    loop {
        match bytes.next() {
            None => break,
            Some(&header) => {
                let num_bytes = ((header & 0x3) + 1) as usize;
                let run_length = ((header >> 2) + 1) as usize;
                let (_, right) = bytes.as_slice().split_at(num_bytes * run_length);
                bytes = right.iter();
                acc += run_length;
            }
        }
    }

    acc
}
