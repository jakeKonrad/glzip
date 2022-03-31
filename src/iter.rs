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

use std::{cmp::Ordering, mem};

//use rand::{
//    distributions::{Distribution, Uniform},
//    Rng,
//};

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

pub struct Union<T, I, J>
where
    T: Ord,
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
{
    x: Option<T>,
    y: Option<T>,
    xs: I,
    ys: J,
}

impl<T, I, J> Iterator for Union<T, I, J>
where
    T: Ord,
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item>
    {
        match mem::replace(&mut self.x, None) {
            Some(a) => match mem::replace(&mut self.y, None) {
                Some(b) => match a.cmp(&b) {
                    Ordering::Less => {
                        self.x = self.xs.next();
                        self.y = Some(b);
                        Some(a)
                    }
                    Ordering::Equal => {
                        self.x = self.xs.next();
                        self.y = self.ys.next();
                        Some(a)
                    }
                    Ordering::Greater => {
                        self.x = Some(a);
                        self.y = self.ys.next();
                        Some(b)
                    }
                },
                None => {
                    self.x = self.xs.next();
                    Some(a)
                }
            },
            None => match mem::replace(&mut self.y, None) {
                Some(b) => {
                    self.y = self.ys.next();
                    Some(b)
                }
                None => None,
            },
        }
    }
}

pub trait IteratorUnionExt<T: Ord>: Iterator<Item = T> + Sized
{
    fn union<J: Iterator<Item = T> + Sized>(self, other: J) -> Union<T, Self, J>;
}

impl<T, I> IteratorUnionExt<T> for I
where
    I: Iterator<Item = T> + Sized,
    T: Ord,
{
    fn union<J: Iterator<Item = T>>(mut self, mut other: J) -> Union<T, Self, J>
    {
        Union {
            x: self.next(),
            y: other.next(),
            xs: self,
            ys: other,
        }
    }
}

/*/// https://en.m.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm
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
}*/

pub struct Flip<T, I: Iterator<Item = (T, T)> + Sized>(I);

impl<T, I> Iterator for Flip<T, I>
where
    I: Iterator<Item = (T, T)>,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item>
    {
        self.0.next().map(|a| (a.1, a.0))
    }
}

pub trait IteratorFlipExt<T>: Iterator<Item = (T, T)> + Sized
{
    fn flip(self) -> Flip<T, Self>;
}

impl<T, I> IteratorFlipExt<T> for I
where
    I: Iterator<Item = (T, T)>,
{
    fn flip(self) -> Flip<T, Self>
    {
        Flip(self)
    }
}
