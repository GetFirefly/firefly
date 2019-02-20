#![cfg_attr(not(test), allow(dead_code))]

pub enum Encoding {
    Latin1,
    Unicode,
    Utf8,
}

pub struct Index(pub usize);

pub struct Table {
    names: Vec<String>,
}

impl Table {
    pub fn new() -> Table {
        Table { names: Vec::new() }
    }

    pub fn str_to_index(&mut self, name: &str) -> Index {
        let existing_position = self
            .names
            .iter()
            .position(|existing_name| existing_name == name);

        let found_or_existing_position = match existing_position {
            Some(position) => position,
            None => {
                self.names.push(name.to_string());
                self.names.len() - 1
            }
        };

        Index(found_or_existing_position)
    }

    pub fn name(&self, index: Index) -> String {
        self.names[index.0].clone()
    }
}
