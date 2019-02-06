#![cfg_attr(not(test), allow(dead_code))]

pub struct Index(pub usize);

pub struct Table {
    names: Vec<String>,
}

impl Table {
    pub fn new() -> Table {
        Table { names: Vec::new() }
    }

    pub fn find_or_insert(&mut self, name: &str) -> Index {
        let found_or_new_index = match self
            .names
            .iter()
            .position(|existing_name| existing_name == name)
        {
            Some(index) => index,
            None => {
                self.names.push(name.to_string());
                self.names.len() - 1
            }
        };

        Index(found_or_new_index)
    }

    pub fn name(&self, index: Index) -> String {
        self.names[index.0].clone()
    }
}
