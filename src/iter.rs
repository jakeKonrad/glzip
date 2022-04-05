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

use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

pub struct Dedup<I: Iterator>
{
    prev: Option<<I as Iterator>::Item>,
    iter: I,
}

impl<I> Iterator for Dedup<I>
where
    I: Iterator,
    <I as Iterator>::Item: PartialEq + Copy,
{
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item>
    {
        loop {
            match self.iter.next() {
                Some(a) => match self.prev {
                    Some(b) => {
                        if a != b {
                            let x = Some(a);
                            self.prev = x;
                            break x;
                        }
                    }
                    None => {
                        let x = Some(a);
                        self.prev = x;
                        break x;
                    }
                },
                None => {
                    break None;
                }
            }
        }
    }
}

pub trait IteratorDedupExt: Iterator + Sized
{
    fn dedup(self) -> Dedup<Self>;
}

impl<I> IteratorDedupExt for I
where
    I: Iterator + Sized,
    <I as Iterator>::Item: PartialEq + Copy,
{
    fn dedup(self) -> Dedup<Self>
    {
        Dedup {
            prev: None,
            iter: self,
        }
    }
}

/// https://en.m.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm
pub trait IteratorReservoirSamplingExt: Iterator + Sized
{
    fn reservoir_sample<R: Rng + ?Sized>(self, rng: &mut R, k: usize) -> Vec<Self::Item>;
}

impl<I> IteratorReservoirSamplingExt for I
where
    I: Iterator + Sized,
{
    fn reservoir_sample<R: Rng + ?Sized>(mut self, rng: &mut R, k: usize) -> Vec<Self::Item>
    {
        let mut buf = Vec::with_capacity(k);

        for _ in 0..k {
            match self.next() {
                Some(x) => buf.push(x),
                None => return buf,
            }
        }

        let indexing_range = Uniform::new(0, k);

        let open_unit_interval = Uniform::new(f64::MIN_POSITIVE, 1.0);

        let k = k as f64;

        let mut w = (open_unit_interval.sample(rng).ln() / k).exp();

        loop {
            let i = (open_unit_interval.sample(rng).ln() / (1.0 - w).ln()).floor() as usize + 1;
            match self.nth(i) {
                Some(x) => {
                    buf[indexing_range.sample(rng)] = x;
                    w *= (open_unit_interval.sample(rng).ln() / k).exp()
                }
                None => break,
            }
        }

        buf
    }
}

