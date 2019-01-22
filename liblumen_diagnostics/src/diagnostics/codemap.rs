use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use itertools::Itertools;

use super::filemap::{FileMap, FileName};
use super::index::{ByteIndex, ByteOffset, RawIndex};

#[derive(Clone, Debug, Default)]
pub struct CodeMap {
    files: Vec<Arc<FileMap>>,
}
impl CodeMap {
    /// Creates an empty `CodeMap`.
    pub fn new() -> CodeMap {
        CodeMap::default()
    }

    /// The next start index to use for a new filemap
    fn next_start_index(&self) -> ByteIndex {
        let end_index = self
            .files
            .last()
            .map(|x| x.span().end())
            .unwrap_or_else(ByteIndex::none);

        // Add one byte of padding between each file
        end_index + ByteOffset(1)
    }

    /// Adds a filemap to the codemap with the given name and source item
    pub fn add_filemap<S>(&mut self, name: FileName, src: S) -> Arc<FileMap>
    where
        S: AsRef<str>,
    {
        let file = Arc::new(FileMap::with_index(
            name,
            src.as_ref().to_owned(),
            self.next_start_index(),
        ));
        self.files.push(file.clone());
        file
    }

    /// Adds a filemap to the codemap with the given name and source string
    pub fn add_filemap_from_string(&mut self, name: FileName, src: String) -> Arc<FileMap> {
        let file = Arc::new(FileMap::with_index(name, src, self.next_start_index()));
        self.files.push(file.clone());
        file
    }

    /// Adds a filemap to the codemap with the given name and source string
    pub fn add_filemap_from_disk<P: Into<PathBuf>>(&mut self, name: P) -> io::Result<Arc<FileMap>> {
        let file = Arc::new(FileMap::from_disk(name, self.next_start_index())?);
        self.files.push(file.clone());
        Ok(file)
    }

    /// Looks up the `File` that contains the specified byte index.
    pub fn find_file(&self, index: ByteIndex) -> Option<&Arc<FileMap>> {
        self.find_index(index).map(|i| &self.files[i])
    }

    pub fn update(&mut self, index: ByteIndex, src: String) -> Option<Arc<FileMap>> {
        self.find_index(index).map(|i| {
            let min = if i == 0 {
                ByteIndex(1)
            } else {
                self.files[i - 1].span().end() + ByteOffset(1)
            };
            let max = self
                .files
                .get(i + 1)
                .map_or(ByteIndex(RawIndex::max_value()), |file_map| {
                    file_map.span().start()
                })
                - ByteOffset(1);
            if src.len() <= (max - min).to_usize() {
                let start_index = self.files[i].span().start();
                let name = self.files[i].name().clone();
                let new_file = Arc::new(FileMap::with_index(name, src, start_index));
                self.files[i] = new_file.clone();
                new_file
            } else {
                let file = self.files.remove(i);
                match self
                    .files
                    .first()
                    .map(|file| file.span().start().to_usize() - 1)
                    .into_iter()
                    .chain(self.files.iter().tuple_windows().map(|(x, y)| {
                        eprintln!("{} {}", x.span(), y.span());
                        (y.span().start() - x.span().end()).to_usize() - 1
                    }))
                    .position(|size| size >= src.len() + 1)
                {
                    Some(j) => {
                        let start_index = if j == 0 {
                            ByteIndex(1)
                        } else {
                            self.files[j - 1].span().end() + ByteOffset(1)
                        };
                        let new_file =
                            Arc::new(FileMap::with_index(file.name().clone(), src, start_index));
                        self.files.insert(j, new_file.clone());
                        new_file
                    }
                    None => self.add_filemap(file.name().clone(), src),
                }
            }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<FileMap>> {
        self.files.iter()
    }

    fn find_index(&self, index: ByteIndex) -> Option<usize> {
        use std::cmp::Ordering;

        self.files
            .binary_search_by(|file| match () {
                () if file.span().start() > index => Ordering::Greater,
                () if file.span().end() < index => Ordering::Less,
                () => Ordering::Equal,
            })
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diagnostics::index::{ByteIndex, RawIndex};
    use crate::diagnostics::span::Span;

    fn check_maps(code_map: &CodeMap, files: &[(RawIndex, &str, &str)]) {
        println!("{:?}", code_map);
        assert_eq!(code_map.files.len(), files.len());
        let mut prev_span = Span::new(0.into(), 0.into());
        for (i, (file, &(start, name, src))) in code_map.files.iter().zip(files).enumerate() {
            println!("{}: {:?} <=> {:?}", i, file, (start, name, src));
            match *file.name() {
                FileName::Virtual(ref virt) => assert_eq!(*virt, name, "At index {}", i),
                _ => panic!(),
            }
            assert_eq!(ByteIndex(start), file.span().start(), "At index {}", i);
            assert!(prev_span.end() < file.span().start(), "At index {}", i);
            assert_eq!(file.src(), src, "At index {}", i);

            prev_span = file.span();
        }
    }

    #[test]
    fn update() {
        let mut code_map = CodeMap::new();

        let a_span = code_map
            .add_filemap_from_string("a".into(), "a".into())
            .span();
        let b_span = code_map
            .add_filemap_from_string("b".into(), "b".into())
            .span();
        let c_span = code_map
            .add_filemap_from_string("c".into(), "c".into())
            .span();

        code_map.update(a_span.start(), "aa".into()).unwrap();
        check_maps(&code_map, &[(3, "b", "b"), (5, "c", "c"), (7, "a", "aa")]);

        code_map.update(b_span.start(), "".into()).unwrap().span();
        check_maps(&code_map, &[(3, "b", ""), (5, "c", "c"), (7, "a", "aa")]);

        code_map.update(c_span.start(), "ccc".into()).unwrap();
        check_maps(&code_map, &[(3, "b", ""), (7, "a", "aa"), (10, "c", "ccc")]);
    }
}
