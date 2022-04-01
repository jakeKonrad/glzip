pub struct ExponentialChunks<'a, T>
{
    data: &'a [T],
    k: usize,
    start: usize,
}

impl<'a, T> Iterator for ExponentialChunks<'a, T>
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.start < self.data.len() {
            let start = self.start;
            let end = start + self.k;
            if end < self.data.len() {
                self.k *= 2;
                self.start = end;
                Some(&self.data[start..end])
            }
            else {
                self.start = self.data.len();
                Some(&self.data[start..])
            }
        }
        else {
            None
        }
    }
}

pub trait SliceExponentialChunksExt<'a, T>
{
    fn exponential_chunks(self, initial_chunk: usize) -> ExponentialChunks<'a, T>;
}

impl<'a, T> SliceExponentialChunksExt<'a, T> for &'a [T]
{
    fn exponential_chunks(self, initial_chunk: usize) -> ExponentialChunks<'a, T>
    {
        ExponentialChunks {
            data: self,
            k: initial_chunk,
            start: 0,
        }
    }
}
