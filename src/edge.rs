
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge(pub u32, pub u32);

impl<T: Into<u32> + Copy> From<[T; 2]> for Edge
{
    fn from(arr: [T; 2]) -> Self
    {
        Self(arr[0].into(), arr[1].into())
    }
}

impl<T: Into<u32> + Copy> From<(T, T)> for Edge
{
    fn from(tup: (T, T)) -> Self
    {
        Self(tup.0.into(), tup.1.into())
    }
}

