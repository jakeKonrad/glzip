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

use std::{
    fs::File,
    io::{BufRead, BufReader, Lines},
    path::Path,
};

use flate2::read::GzDecoder;

use super::{raise, BoxedError};
use crate::Edge;

fn parse_edge(line: String) -> Result<Edge, BoxedError>
{
    let mut iter = line.split(',');
    let x = iter
        .next()
        .ok_or_else(|| raise(format!("bad row: {}", line.clone())))?;
    let y = iter
        .next()
        .ok_or_else(|| raise(format!("bad row: {}", line.clone())))?;
    iter.next()
        .map_or_else(|| Ok(()), |_| Err(format!("bad row: {}", line.clone())))?;
    let x = x.trim().parse::<u32>()?;
    let y = y.trim().parse::<u32>()?;
    Ok([x, y])
}

pub(super) struct Csv(Lines<BufReader<File>>);

impl Csv
{
    fn new(file: File) -> Self
    {
        Self(BufReader::new(file).lines())
    }
}

impl Iterator for Csv
{
    type Item = Result<Edge, BoxedError>;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.0.next().map(|s| {
            let s = s?;
            parse_edge(s)
        })
    }
}

pub(super) struct CsvGz(Lines<BufReader<GzDecoder<File>>>);

impl CsvGz
{
    fn new(file: File) -> Self
    {
        Self(BufReader::new(GzDecoder::new(file)).lines())
    }
}

impl Iterator for CsvGz
{
    type Item = Result<Edge, BoxedError>;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.0.next().map(|s| {
            let s = s?;
            parse_edge(s)
        })
    }
}

pub(super) enum Iter
{
    Csv(Csv),
    CsvGz(CsvGz),
}

impl Iterator for Iter
{
    type Item = Result<Edge, BoxedError>;

    fn next(&mut self) -> Option<Self::Item>
    {
        match self {
            Self::Csv(iter) => iter.next(),
            Self::CsvGz(iter) => iter.next(),
        }
    }
}

pub(super) fn load<P: AsRef<Path>>(p: P) -> Result<Iter, BoxedError>
{
    let path = p.as_ref();
    path.extension().map_or_else(
        || {
            Err(raise(format!(
                "bad path: {:?}",
                path.as_os_str().to_os_string()
            )))
        },
        |ext| {
            if ext == "gz" {
                path.file_stem()
                    .map(Path::new)
                    .and_then(|stem| stem.extension())
                    .map_or_else(
                        || {
                            Err(raise(format!(
                                "bad path: {:?}",
                                path.as_os_str().to_os_string()
                            )))
                        },
                        |ext2| {
                            if ext2 == "csv" {
                                let f = File::open(path)?;
                                Ok(Iter::CsvGz(CsvGz::new(f)))
                            }
                            else {
                                Err(raise(format!(
                                    "bad path: {:?}",
                                    path.as_os_str().to_os_string()
                                )))
                            }
                        },
                    )
            }
            else if ext == "csv" {
                let f = File::open(path)?;
                Ok(Iter::Csv(Csv::new(f)))
            }
            else {
                Err(raise(format!(
                    "bad path: {:?}",
                    path.as_os_str().to_os_string()
                )))
            }
        },
    )
}
