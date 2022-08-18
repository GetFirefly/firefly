use std::fmt;

use rpds::{RedBlackTreeMap, RedBlackTreeSet};

use liblumen_intern::{symbols, Ident, Symbol};

use crate::{Literal, Type};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Annotation {
    // The common case is just a symbol with significant meaning
    Unit,
    // In some cases we need more flexibility, so we use literals for that purpose,
    // with a key used to unique the annotations for an expression
    Term(Literal),
    // Used for tracking used/new variables associated with expressions
    Vars(RedBlackTreeSet<Ident>),
    // Used for tracking the type associated with an item
    Type(Type),
}
impl From<Literal> for Annotation {
    #[inline]
    fn from(term: Literal) -> Self {
        Self::Term(term)
    }
}
impl From<Type> for Annotation {
    #[inline]
    fn from(ty: Type) -> Self {
        Self::Type(ty)
    }
}
impl From<RedBlackTreeSet<Ident>> for Annotation {
    #[inline]
    fn from(set: RedBlackTreeSet<Ident>) -> Self {
        Self::Vars(set)
    }
}

#[derive(Clone, PartialEq, Eq, Default)]
pub struct Annotations(RedBlackTreeMap<Symbol, Annotation>);
unsafe impl Send for Annotations {}
unsafe impl Sync for Annotations {}
impl fmt::Debug for Annotations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        f.write_str("@annotations[")?;
        for (key, anno) in self.0.iter() {
            crate::printing::print_annotation(f, key, anno)?;
        }
        f.write_char(']')
    }
}
impl Annotations {
    /// Create a new, empty annotation set
    pub fn new() -> Self {
        Self(RedBlackTreeMap::new())
    }

    /// Creates a new annotation set initialized with symbols::CompilerGenerated
    pub fn default_compiler_generated() -> Self {
        let mut this = Self::default();
        this.insert_mut(symbols::CompilerGenerated, Annotation::Unit);
        this
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &Annotation)> {
        self.0.iter()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Clear all annotations
    #[inline]
    pub fn clear(&mut self) {
        self.0 = RedBlackTreeMap::new();
    }

    /// Pushes a new annotation on the set
    #[inline]
    pub fn insert<A: Into<Annotation>>(&self, key: Symbol, anno: A) -> Self {
        Self(self.0.insert(key, anno.into()))
    }

    #[inline]
    pub fn insert_mut<A: Into<Annotation>>(&mut self, key: Symbol, anno: A) {
        self.0.insert_mut(key, anno.into());
    }

    /// Pushes a new unit annotation on the set
    #[inline]
    pub fn set(&mut self, key: Symbol) {
        self.0.insert_mut(key, Annotation::Unit);
    }

    /// Tests if the given annotation key is present in the set
    #[inline]
    pub fn contains(&self, key: Symbol) -> bool {
        self.0.contains_key(&key)
    }

    /// Retrieves an annotation by key
    #[inline]
    pub fn get(&self, key: Symbol) -> Option<&Annotation> {
        self.0.get(&key)
    }

    /// Retrieves an annotation by key, mutably
    #[inline]
    pub fn get_mut(&mut self, key: Symbol) -> Option<&mut Annotation> {
        self.0.get_mut(&key)
    }

    /// Removes an annotation with the given key
    #[inline]
    pub fn remove(&self, key: Symbol) -> Self {
        Self(self.0.remove(&key))
    }

    /// Removes an annotation with the given key
    #[inline]
    pub fn remove_mut(&mut self, key: Symbol) {
        self.0.remove_mut(&key);
    }

    /// Replaces all annotations with the given set
    pub fn replace(&mut self, annotations: Self) {
        self.0 = annotations.0;
    }

    /// Convenience function for accessing the new vars annotation
    pub fn new_vars(&self) -> Option<RedBlackTreeSet<Ident>> {
        match self.get(symbols::New) {
            None => None,
            Some(Annotation::Vars(set)) => Some(set.clone()),
            Some(_) => None,
        }
    }

    /// Convenience function for accessing the used vars annotation
    pub fn used_vars(&self) -> Option<RedBlackTreeSet<Ident>> {
        match self.get(symbols::Used) {
            None => None,
            Some(Annotation::Vars(set)) => Some(set.clone()),
            Some(_) => None,
        }
    }
}
impl From<Symbol> for Annotations {
    fn from(sym: Symbol) -> Self {
        let mut annos = Self::default();
        annos.set(sym);
        annos
    }
}
impl<const N: usize> From<[Symbol; N]> for Annotations {
    #[inline]
    fn from(syms: [Symbol; N]) -> Self {
        Self::from(syms.as_slice())
    }
}
impl<const N: usize> From<[(Symbol, Annotation); N]> for Annotations {
    #[inline]
    fn from(syms: [(Symbol, Annotation); N]) -> Self {
        Self::from(syms.as_slice())
    }
}
impl From<&[Symbol]> for Annotations {
    fn from(syms: &[Symbol]) -> Self {
        let mut annos = Self::default();
        for sym in syms {
            annos.set(*sym);
        }
        annos
    }
}
impl From<&[(Symbol, Annotation)]> for Annotations {
    fn from(syms: &[(Symbol, Annotation)]) -> Self {
        let mut annos = Self::default();
        for (sym, anno) in syms {
            annos.insert_mut(*sym, anno.clone());
        }
        annos
    }
}
impl From<Vec<(Symbol, Annotation)>> for Annotations {
    fn from(mut annos: Vec<(Symbol, Annotation)>) -> Self {
        Self(annos.drain(..).collect())
    }
}

/// This trait is implemented by all types which carry annotations
pub trait Annotated {
    fn annotations(&self) -> &Annotations;
    fn annotations_mut(&mut self) -> &mut Annotations;
    #[inline]
    fn annotate<A: Into<Annotation>>(&mut self, key: Symbol, anno: A) {
        self.annotations_mut().insert_mut(key, anno);
    }
    /// Returns true if this item has been marked as generated by the compiler
    #[inline]
    fn is_compiler_generated(&self) -> bool {
        self.annotations().contains(symbols::CompilerGenerated)
    }
    /// Sets the symbols::CompilerGenerated annotation on this item
    #[inline]
    fn mark_compiler_generated(&mut self) {
        self.annotations_mut().set(symbols::CompilerGenerated);
    }
    /// Returns true if this item has an annotation with the given key
    #[inline]
    fn has_annotation(&self, name: Symbol) -> bool {
        self.annotations().contains(name)
    }
    /// Gets the currently known type of this item.
    ///
    /// If the type was not previously set, this returns Type::Unknown
    #[inline]
    fn get_type(&self) -> Type {
        self.annotations()
            .get(symbols::Type)
            .map(|anno| match anno {
                Annotation::Type(ty) => ty.clone(),
                _ => Type::default(),
            })
            .unwrap_or_default()
    }
    /// Sets the type of this item to the given type
    #[inline]
    fn set_type(&mut self, ty: Type) {
        self.annotations_mut().insert_mut(symbols::Type, ty)
    }
    #[inline]
    fn used_vars(&self) -> RedBlackTreeSet<Ident> {
        self.annotations()
            .used_vars()
            .map(|v| v.clone())
            .unwrap_or_default()
    }
    #[inline]
    fn new_vars(&self) -> RedBlackTreeSet<Ident> {
        self.annotations()
            .new_vars()
            .map(|v| v.clone())
            .unwrap_or_default()
    }
}
impl<T> Annotated for Box<T>
where
    T: Annotated,
{
    fn annotations(&self) -> &Annotations {
        (&**self).annotations()
    }
    fn annotations_mut(&mut self) -> &mut Annotations {
        (&mut **self).annotations_mut()
    }
}
