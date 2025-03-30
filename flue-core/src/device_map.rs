#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceMap {
    ForceCpu,
    Ordinal(usize),
}

impl Default for DeviceMap {
    fn default() -> Self {
        Self::Ordinal(0)
    }
}
