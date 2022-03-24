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

use std::{mem::MaybeUninit, ops::Range, slice::Iter};

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

pub struct Group
{
    range: Range<usize>,
    data: [MaybeUninit<u32>; 64],
}

impl Iterator for Group
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item>
    {
        let i = self.range.next()?;
        Some(unsafe { self.data[i].assume_init() })
    }
}

fn next_group<'a>(prev_edge: &mut u32, bytes: &mut Iter<'a, u8>) -> Option<Group>
{
    match bytes.next() {
        None => None,
        Some(&header) => {
            let num_bytes = ((header & 0x3) + 1) as usize;
            let run_length = ((header >> 2) + 1) as usize;
            let (left, right) = bytes.as_slice().split_at(num_bytes * run_length);
            *bytes = right.iter();

            let mut buf: [MaybeUninit<u32>; 64] = unsafe { MaybeUninit::uninit().assume_init() };

            match num_bytes {
                1 => {
                    for (i, chunk) in left.chunks_exact(1).enumerate() {
                        buf[i].write(chunk[0] as u32);
                    }
                }
                2 => {
                    for (i, chunk) in left.chunks_exact(2).enumerate() {
                        let mut diff = (chunk[0] as u32) << 8;
                        diff |= chunk[1] as u32;
                        buf[i].write(diff);
                    }
                }
                3 => {
                    for (i, chunk) in left.chunks_exact(3).enumerate() {
                        let mut diff = (chunk[0] as u32) << 16;
                        diff |= (chunk[1] as u32) << 8;
                        diff |= chunk[2] as u32;
                        buf[i].write(diff);
                    }
                }
                4 => {
                    for (i, chunk) in left.chunks_exact(4).enumerate() {
                        let mut diff = (chunk[0] as u32) << 24;
                        diff |= (chunk[1] as u32) << 16;
                        diff |= (chunk[2] as u32) << 8;
                        diff |= chunk[3] as u32;
                        buf[i].write(diff);
                    }
                }
                _ => unreachable!(),
            }

            let mut p_edge = *prev_edge;
            for item in buf.iter_mut().take(run_length) {
                let nnz = unsafe { item.assume_init() };
                let sum = p_edge + nnz;
                p_edge = sum;
                item.write(sum);
            }
            *prev_edge = p_edge;

            Some(Group {
                range: 0..run_length,
                data: buf,
            })
        }
    }
}

pub enum Decoder<'a>
{
    FirstEdge(u32, Iter<'a, u8>),
    NextEdge(Group, u32, Iter<'a, u8>),
}

impl<'a> Iterator for Decoder<'a>
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self {
            Self::FirstEdge(source, iter) => match first_edge(*source, iter) {
                Some(e) => {
                    let mut prev_edge = e;
                    match next_group(&mut prev_edge, iter) {
                        Some(group) => {
                            *self = Self::NextEdge(group, prev_edge, iter.clone());
                            Some(e)
                        }
                        None => Some(e),
                    }
                }
                None => None,
            },
            Self::NextEdge(group, prev_edge, iter) => match group.next() {
                Some(e) => Some(e),
                None => match next_group(prev_edge, iter) {
                    Some(mut group) => {
                        let res = group.next();
                        *self = Self::NextEdge(group, *prev_edge, iter.clone());
                        res
                    }
                    None => None,
                },
            },
        }
    }
}

pub fn decode(source: u32, bytes: &[u8]) -> Decoder<'_>
{
    Decoder::FirstEdge(source, bytes.iter())
}

pub fn count(source: u32, bytes: &[u8]) -> usize
{
    let mut bytes = bytes.iter();

    first_edge(source, &mut bytes);

    let mut acc = 0;

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
