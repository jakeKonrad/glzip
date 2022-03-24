use std::{convert::TryInto, error, fmt};

pub enum Error<Ix, T>
where
    Ix: TryInto<usize>,
    T: TryInto<u32>,
{
    TryIntoUsize(<Ix as TryInto<usize>>::Error),
    TryIntoU32(<T as TryInto<u32>>::Error),
    TooManyVertices,
}

impl<Ix, T> fmt::Debug for Error<Ix, T>
where
    Ix: TryInto<usize>,
    <Ix as TryInto<usize>>::Error: fmt::Debug,
    T: TryInto<u32>,
    <T as TryInto<u32>>::Error: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self {
            Self::TryIntoUsize(ix_error) => ix_error.fmt(f),
            Self::TryIntoU32(t_error) => t_error.fmt(f),
            Self::TooManyVertices => write!(f, "Error::TooManyVertices"),
        }
    }
}

impl<Ix, T> fmt::Display for Error<Ix, T>
where
    Ix: TryInto<usize>,
    <Ix as TryInto<usize>>::Error: fmt::Display,
    T: TryInto<u32>,
    <T as TryInto<u32>>::Error: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self {
            Self::TryIntoUsize(ix_error) => ix_error.fmt(f),
            Self::TryIntoU32(t_error) => t_error.fmt(f),
            Self::TooManyVertices => write!(
                f,
                "too many vertices, cannot handle more than {} vertices",
                u32::MAX
            ),
        }
    }
}

impl<Ix, T> error::Error for Error<Ix, T>
where
    Ix: TryInto<usize>,
    <Ix as TryInto<usize>>::Error: error::Error,
    T: TryInto<u32>,
    <T as TryInto<u32>>::Error: error::Error,
{
}

pub enum Void {}

impl From<Void> for usize
{
    fn from(void: Void) -> usize
    {
        match void {}
    }
}

impl From<Void> for u32
{
    fn from(void: Void) -> u32
    {
        match void {}
    }
}
