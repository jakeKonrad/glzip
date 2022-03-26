#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct Edge(pub u32, pub u32);

impl<T: Into<u32>> From<[T; 2]> for Edge
{
    fn from(arr: [T; 2]) -> Self
    {
        let [u, v] = arr;
        Self(u.into(), v.into())
    }
}

impl<T: Into<u32>> From<(T, T)> for Edge
{
    fn from(tup: (T, T)) -> Self
    {
        let (u, v) = tup;
        Self(u.into(), v.into())
    }
}

impl From<Edge> for (u32, u32)
{
    fn from(edge: Edge) -> Self
    {
        (edge.0, edge.1)
    }
}

impl From<Edge> for [u32; 2]
{
    fn from(edge: Edge) -> Self
    {
        [edge.0, edge.1]
    }
}
