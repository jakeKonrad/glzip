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
