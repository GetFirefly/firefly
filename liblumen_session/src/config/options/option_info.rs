/// Represents metadata about an option
#[derive(Debug)]
pub struct OptionInfo {
    pub name: &'static str,
    pub description: Option<&'static str>,
}
impl OptionInfo {
    pub const fn from_name(name: &'static str) -> Self {
        Self {
            name,
            description: None,
        }
    }

    pub const fn new(name: &'static str, description: Option<&'static str>) -> Self {
        Self { name, description }
    }
}
