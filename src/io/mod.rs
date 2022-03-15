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

use std::{error::Error, path::Path};

use crate::Edge;

#[cfg(feature = "csv")]
mod csv;

type BoxedError = Box<dyn Error + Send + Sync>;

fn raise<S: AsRef<str>>(payload: S) -> BoxedError
{
    Box::from(payload.as_ref())
}

enum Iter
{
    #[cfg(feature = "csv")]
    Csv(csv::Iter),
}

impl Iterator for Iter
{
    type Item = Result<Edge, BoxedError>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self {
            Iter::Csv(iter) => iter.next(),
        }
    }
}

pub fn load<P: AsRef<Path>>(
    path: P,
) -> Result<impl Iterator<Item = Result<Edge, BoxedError>>, BoxedError>
{
    if cfg!(feature = "csv") {
        csv::load(path).map(|csv| Iter::Csv(csv))
    }
    else {
        unreachable!()
    }
}
