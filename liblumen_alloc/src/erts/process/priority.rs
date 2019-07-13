#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Max,
}

impl Default for Priority {
    fn default() -> Priority {
        Priority::Normal
    }
}
