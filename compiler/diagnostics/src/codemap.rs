use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use super::*;

#[derive(Debug)]
pub struct CodeMap {
    files: DashMap<SourceId, Arc<SourceFile>>,
    names: DashMap<FileName, SourceId>,
    seen: DashMap<PathBuf, SourceId>,
    next_file_id: AtomicU32,
}
impl CodeMap {
    /// Creates an empty `CodeMap`.
    pub fn new() -> Self {
        Self {
            files: DashMap::new(),
            names: DashMap::new(),
            seen: DashMap::new(),
            next_file_id: AtomicU32::new(1),
        }
    }

    /// Add a file to the map, returning the handle that can be used to
    /// refer to it again.
    pub fn add(&self, name: impl Into<FileName>, source: String) -> SourceId {
        // De-duplicate real files on add; it _may_ be possible for concurrent
        // adds to add the same file more than once, since we're working across
        // two maps; but since DashMap uses read/write locks internally to lock
        // buckets, the sequence of locks required here should prevent that from
        // happening
        //
        // We don't de-duplicate virtual files, because the same name could be used
        // for different content, and its unlikely that we'd be adding the same content
        // over and over again with the same virtual file name
        let name = name.into();
        if let FileName::Real(ref path) = name {
            let seen_ref = self
                .seen
                .entry(path.clone())
                .or_insert_with(|| self.insert_file(name, source, None));
            *seen_ref.value()
        } else {
            self.insert_file(name, source, None)
        }
    }

    /// Add a file to the map with the given source span as a parent.
    /// This will not deduplicate the file in the map.
    pub fn add_child(
        &self,
        name: impl Into<FileName>,
        source: String,
        parent: SourceSpan,
    ) -> SourceId {
        self.insert_file(name.into(), source, Some(parent))
    }

    fn insert_file(&self, name: FileName, source: String, parent: Option<SourceSpan>) -> SourceId {
        let file_id = self.next_file_id();
        let filename = name.clone();
        self.files.insert(
            file_id,
            Arc::new(SourceFile::new(file_id, name.into(), source, parent)),
        );
        self.names.insert(filename, file_id);
        file_id
    }

    /// Get the file corresponding to the given id.
    pub fn get(&self, file_id: SourceId) -> Result<Arc<SourceFile>, Error> {
        if file_id == SourceId::UNKNOWN {
            Err(Error::FileMissing)
        } else {
            self.files
                .get(&file_id)
                .map(|r| r.value().clone())
                .ok_or(Error::FileMissing)
        }
    }

    /// Get the file corresponding to the given SourceSpan
    pub fn get_with_span(&self, span: SourceSpan) -> Result<Arc<SourceFile>, Error> {
        self.get(span.source_id)
    }

    pub fn parent(&self, file_id: SourceId) -> Option<SourceSpan> {
        self.get(file_id).ok().and_then(|f| f.parent())
    }

    /// Get the file id corresponding to the given FileName
    pub fn get_file_id(&self, filename: &FileName) -> Option<SourceId> {
        self.names.get(filename).map(|id| *id)
    }

    /// Get the file corresponding to the given FileName
    pub fn get_by_name(&self, filename: &FileName) -> Option<Arc<SourceFile>> {
        self.get_file_id(filename).and_then(|id| self.get(id).ok())
    }

    /// Get the filename corresponding to the given SourceId
    pub fn name(&self, file_id: SourceId) -> Result<FileName, Error> {
        let file = self.get(file_id)?;
        Ok(file.name().clone())
    }

    /// Get the filename associated with the given SourceSpan
    pub fn name_for_span(&self, span: SourceSpan) -> Result<FileName, Error> {
        self.name(span.source_id)
    }

    /// Get the filename associated with the given Spanned item
    pub fn name_for_spanned<T>(&self, spanned: Spanned<T>) -> Result<FileName, Error> {
        self.name(spanned.span.source_id)
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Arc<SourceFile>> + 'a {
        self.files.iter().map(|r| r.value().clone())
    }

    pub fn line_span(
        &self,
        file_id: SourceId,
        line_index: impl Into<LineIndex>,
    ) -> Result<Span, Error> {
        let f = self.get(file_id)?;
        f.line_span(line_index.into())
    }

    pub fn line_index(
        &self,
        file_id: SourceId,
        byte_index: impl Into<ByteIndex>,
    ) -> Result<LineIndex, Error> {
        Ok(self.get(file_id)?.line_index(byte_index.into()))
    }

    pub fn location(
        &self,
        file_id: SourceId,
        byte_index: impl Into<ByteIndex>,
    ) -> Result<Location, Error> {
        self.get(file_id)?.location(byte_index)
    }

    /// Get the Location associated with the given SourceSpan
    pub fn location_for_span(&self, span: SourceSpan) -> Result<Location, Error> {
        self.location(span.source_id, span)
    }

    /// Get the Location associated with the given Spanned item
    pub fn location_for_spanned<T>(&self, spanned: &Spanned<T>) -> Result<Location, Error> {
        self.location(spanned.span.source_id, spanned.span)
    }

    pub fn source_span(&self, file_id: SourceId) -> Result<SourceSpan, Error> {
        Ok(self.get(file_id)?.source_span())
    }

    pub fn source_slice<'a>(
        &'a self,
        file_id: SourceId,
        span: impl Into<Span>,
    ) -> Result<&'a str, Error> {
        let f = self.get(file_id)?;
        let slice = f.source_slice(span.into())?;
        unsafe { Ok(std::mem::transmute::<&str, &'a str>(slice)) }
    }

    /// Get the source string associated with the given Spanned item
    pub fn source_slice_for_spanned<'a, T>(
        &'a self,
        spanned: &Spanned<T>,
    ) -> Result<&'a str, Error> {
        self.source_slice(spanned.span.source_id, spanned.span)
    }

    #[inline(always)]
    fn next_file_id(&self) -> SourceId {
        let id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        SourceId::new(id)
    }
}
impl Default for CodeMap {
    fn default() -> Self {
        Self::new()
    }
}
impl<'a> Files<'a> for CodeMap {
    type FileId = SourceId;
    type Name = String;
    type Source = &'a str;

    fn name(&self, file_id: Self::FileId) -> Result<Self::Name, Error> {
        Ok(format!("{}", self.get(file_id)?.name()))
    }

    fn source(&self, file_id: Self::FileId) -> Result<&'a str, Error> {
        use std::mem;

        let f = self.get(file_id)?;
        Ok(unsafe { mem::transmute::<&str, &'a str>(f.source()) })
    }

    fn line_index(&self, file_id: Self::FileId, byte_index: usize) -> Result<usize, Error> {
        Ok(self.line_index(file_id, byte_index as u32)?.to_usize())
    }

    fn line_range(&self, file_id: Self::FileId, line_index: usize) -> Result<Range<usize>, Error> {
        let span = self.line_span(file_id, line_index as u32)?;

        Ok(span.start().to_usize()..span.end().to_usize())
    }
}
