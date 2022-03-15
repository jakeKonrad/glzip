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

use std::{iter, iter::Peekable};

fn first_edge(bytes: &mut Vec<u8>, source: u32, target: u32)
{
    let mut diff = target.abs_diff(source);
    let mut first_byte = (diff as u8) & 0x3f;
    let mut cur_byte: Option<u8> = None;
    bytes.extend(iter::from_fn(move || match cur_byte {
        None => {
            if target < source {
                first_byte |= 0x40;
            }
            diff >>= 6;
            if diff > 0 {
                first_byte |= 0x80;
            }
            cur_byte = Some((diff as u8) & 0x7f);
            Some(first_byte)
        }
        Some(b) if (b > 0) | (diff > 0) => {
            let mut next_byte = b;
            diff >>= 7;
            cur_byte = Some((diff as u8) & 0x7f);
            if diff > 0 {
                next_byte |= 0x80;
            }
            Some(next_byte)
        }
        _ => None,
    }));
}

const ONE_BYTE: u32 = 256; // 0xff + 1
const TWO_BYTES: u32 = 65536; // 0xffff + 1
const THREE_BYTES: u32 = 16777216; // 0xffffff + 1

fn next_group<I>(bytes: &mut Vec<u8>, diffs: &mut Peekable<I>)
where
    I: Iterator<Item = u32>,
{
    match diffs.peek() {
        None => {}
        Some(&diff) if diff < ONE_BYTE => {
            let mut j = 0usize;
            let mut buf = [0u32; 64];
            for i in 0..64 {
                match diffs.next_if(|&d| d < ONE_BYTE) {
                    Some(d) => {
                        j = i;
                        buf[i] = d;
                    }
                    None => break,
                }
            }
            bytes.push((j as u8) << 2);
            for i in 0..=j {
                bytes.push(buf[i] as u8);
            }
        }
        Some(&diff) if diff < TWO_BYTES => {
            let mut j = 0usize;
            let mut buf = [0u32; 64];
            for i in 0..64 {
                match diffs.next_if(|d| (ONE_BYTE..TWO_BYTES).contains(d)) {
                    Some(d) => {
                        j = i;
                        buf[i] = d;
                    }
                    None => break,
                }
            }
            bytes.push(1u8 | ((j as u8) << 2));
            for i in 0..=j {
                let d = buf[i];
                bytes.push((d >> 8) as u8);
                bytes.push(d as u8);
            }
        }
        Some(&diff) if diff < THREE_BYTES => {
            let mut j = 0usize;
            let mut buf = [0u32; 64];
            for i in 0..64 {
                match diffs.next_if(|d| (TWO_BYTES..THREE_BYTES).contains(d)) {
                    Some(d) => {
                        j = i;
                        buf[i] = d;
                    }
                    None => break,
                }
            }
            bytes.push(2u8 | ((j as u8) << 2));
            for i in 0..=j {
                let d = buf[i];
                bytes.push((d >> 16) as u8);
                bytes.push((d >> 8) as u8);
                bytes.push(d as u8);
            }
        }
        Some(_) => {
            let mut j = 0usize;
            let mut buf = [0u32; 64];
            for i in 0..64 {
                match diffs.next_if(|&d| d >= THREE_BYTES) {
                    Some(d) => {
                        j = i;
                        buf[i] = d;
                    }
                    None => break,
                }
            }
            bytes.push(3u8 | ((j as u8) << 2));
            for i in 0..=j {
                let d = buf[i];
                bytes.push((d >> 24) as u8);
                bytes.push((d >> 16) as u8);
                bytes.push((d >> 8) as u8);
                bytes.push(d as u8);
            }
        }
    }
}

struct Diffs<I: Iterator<Item = u32>>
{
    iter: Peekable<I>,
}

impl<I: Iterator<Item = u32>> Diffs<I>
{
    fn new(iter: I) -> Self
    {
        Diffs {
            iter: iter.peekable(),
        }
    }
}

impl<I: Iterator<Item = u32>> Iterator for Diffs<I>
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item>
    {
        let x = self.iter.next()?;
        let &y = self.iter.peek()?;
        assert!(y >= x);
        Some(y - x)
    }
}

pub fn encode<I>(bytes: &mut Vec<u8>, source: u32, into_edges: I)
where
    I: IntoIterator<Item = u32>,
{
    let mut edges = into_edges.into_iter().peekable();

    let edge = match edges.peek() {
        Some(&e) => e,
        None => return,
    };

    first_edge(bytes, source, edge);

    let mut diffs = Diffs::new(edges).peekable();

    while diffs.peek().is_some() {
        next_group(bytes, &mut diffs);
    }
}
