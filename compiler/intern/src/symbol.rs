//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e., given a value, one can easily find the
//! type, and vice versa.
#![allow(unused)]

use std::cell::RefCell;
use std::cmp::{Ord, Ordering, PartialEq, PartialOrd};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::str;
use std::sync::{Arc, RwLock};

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::arena::DroplessArena;
use crate::symbols;

use liblumen_diagnostics::{SourceSpan, Spanned};

lazy_static! {
    /// A globally accessible symbol table
    pub static ref SYMBOL_TABLE: SymbolTable = {
        SymbolTable::new()
    };
}

pub struct SymbolTable {
    interner: RwLock<Interner>,
}
impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            interner: RwLock::new(Interner::fresh()),
        }
    }
}
unsafe impl Sync for SymbolTable {}

#[derive(Copy, Clone, Eq, Spanned)]
pub struct Ident {
    pub name: Symbol,
    #[span]
    pub span: SourceSpan,
}
impl Default for Ident {
    fn default() -> Self {
        Self {
            name: symbols::Empty,
            span: SourceSpan::UNKNOWN,
        }
    }
}
impl Ident {
    #[inline]
    pub const fn new(name: Symbol, span: SourceSpan) -> Ident {
        Ident { name, span }
    }

    #[inline]
    pub const fn with_empty_span(name: Symbol) -> Ident {
        Ident::new(name, SourceSpan::UNKNOWN)
    }

    /// Maps an interned string to an identifier with an empty syntax context.
    pub fn from_interned_str(string: InternedString) -> Ident {
        Ident::with_empty_span(string.as_symbol())
    }

    /// Maps a string to an identifier with an empty syntax context.
    pub fn from_str(string: &str) -> Ident {
        Ident::with_empty_span(Symbol::intern(string))
    }

    pub fn unquote_string(self) -> Ident {
        Ident::new(Symbol::intern(self.as_str().trim_matches('"')), self.span)
    }

    pub fn unquote_atom(self) -> Ident {
        Ident::new(Symbol::intern(self.as_str().trim_matches('\'')), self.span)
    }

    pub fn gensym(self) -> Ident {
        Ident::new(self.name.gensymed(), self.span)
    }

    pub fn as_str(self) -> LocalInternedString {
        self.name.as_str()
    }

    pub fn as_interned_str(self) -> InternedString {
        self.name.as_interned_str()
    }

    pub fn is_keyword(self) -> bool {
        self.name.is_keyword()
    }

    pub fn is_reserved_attr(self) -> bool {
        self.name.is_reserved_attr()
    }

    pub fn is_preprocessor_directive(self) -> bool {
        self.name.is_preprocessor_directive()
    }
}

impl Ord for Ident {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(&other.as_str())
    }
}
impl PartialOrd for Ident {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Ident {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
impl PartialEq<Symbol> for Ident {
    fn eq(&self, rhs: &Symbol) -> bool {
        self.name.eq(rhs)
    }
}
impl Hash for Ident {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ident<{} {:?}>", self.name, self.span)
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolIndex(u32);
impl Clone for SymbolIndex {
    fn clone(&self) -> Self {
        *self
    }
}
impl From<SymbolIndex> for u32 {
    #[inline]
    fn from(v: SymbolIndex) -> u32 {
        v.as_u32()
    }
}
impl From<SymbolIndex> for usize {
    #[inline]
    fn from(v: SymbolIndex) -> usize {
        v.as_usize()
    }
}
impl SymbolIndex {
    // shave off 256 indices at the end to allow space for packing these indices into enums
    pub const MAX_AS_U32: u32 = 0xFFFF_FF00;

    pub const MAX: SymbolIndex = SymbolIndex::new(0xFFFF_FF00);

    #[inline]
    const fn new(n: u32) -> Self {
        // This will fail at const eval time unless `value <=
        // max` is true (in which case we get the index 0).
        // It will also fail at runtime, of course, but in a
        // kind of wacky way.
        let _ = ["out of range value used"][!(n <= Self::MAX_AS_U32) as usize];

        SymbolIndex(n)
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// A symbol is an interned or gensymed string.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(SymbolIndex);

impl Symbol {
    pub const fn new(n: u32) -> Self {
        Symbol(SymbolIndex::new(n))
    }

    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        with_interner(|interner| interner.intern(string))
    }

    pub fn interned(self) -> Self {
        with_interner(|interner| interner.interned(self))
    }

    /// Gensyms a new usize, using the current interner.
    pub fn gensym(string: &str) -> Self {
        with_interner(|interner| interner.gensym(string))
    }

    pub fn gensymed(self) -> Self {
        with_interner(|interner| interner.gensymed(self))
    }

    pub fn as_str(self) -> LocalInternedString {
        with_interner(|interner| unsafe {
            LocalInternedString {
                string: ::std::mem::transmute::<&str, &str>(interner.get(self)),
                dummy: PhantomData,
            }
        })
    }

    pub fn as_interned_str(self) -> InternedString {
        with_interner(|interner| InternedString {
            symbol: interner.interned(self),
        })
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0.as_u32()
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0.as_usize()
    }

    #[inline]
    pub fn is_boolean(&self) -> bool {
        // Booleans are always 0 or 1 by index
        self.0.as_u32() < 2
    }

    /// Returns `true` if the token is a keyword, reserved in all name positions
    #[inline]
    pub fn is_keyword(self) -> bool {
        symbols::is_keyword(self)
    }

    /// Returns `true` if the token is a reserved attribute name
    #[inline]
    pub fn is_reserved_attr(self) -> bool {
        symbols::is_reserved(self)
    }

    /// Returns `true` if the token is a preprocessor directive name
    #[inline]
    pub fn is_preprocessor_directive(self) -> bool {
        symbols::is_directive(self)
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let is_gensymed = with_interner(|interner| interner.is_gensymed(*self));
        if is_gensymed {
            write!(f, "{}({:?})", self, self.0)
        } else {
            write!(f, "{}({:?})", self, self.0)
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.as_str(), f)
    }
}

impl<T: ::std::ops::Deref<Target = str>> PartialEq<T> for Symbol {
    fn eq(&self, other: &T) -> bool {
        self.as_str() == other.deref()
    }
}

// The `&'static str`s in this type actually point into the arena.
//
// Note that normal symbols are indexed upward from 0, and gensyms are indexed
// downward from SymbolIndex::MAX_AS_U32.
#[derive(Default)]
pub struct Interner {
    arena: DroplessArena,
    pub names: FxHashMap<&'static str, Symbol>,
    pub strings: Vec<&'static str>,
    gensyms: Vec<Symbol>,
}

impl Interner {
    pub fn fresh() -> Self {
        let mut this = Interner::default();
        for (sym, s) in symbols::__SYMBOLS {
            this.names.insert(s, *sym);
            this.strings.push(s);
        }
        this
    }

    pub fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        let name = Symbol::new(self.strings.len() as u32);

        // `from_utf8_unchecked` is safe since we just allocated a `&str` which is known to be
        // UTF-8.
        let string: &str =
            unsafe { str::from_utf8_unchecked(self.arena.alloc_slice(string.as_bytes())) };
        // It is safe to extend the arena allocation to `'static` because we only access
        // these while the arena is still alive.
        let string: &'static str = unsafe { &*(string as *const str) };
        self.strings.push(string);
        self.names.insert(string, name);
        name
    }

    pub fn interned(&self, symbol: Symbol) -> Symbol {
        if (symbol.0.as_usize()) < self.strings.len() {
            symbol
        } else {
            self.interned(self.gensyms[(SymbolIndex::MAX_AS_U32 - symbol.0.as_u32()) as usize])
        }
    }

    fn gensym(&mut self, string: &str) -> Symbol {
        let symbol = self.intern(string);
        self.gensymed(symbol)
    }

    fn gensymed(&mut self, symbol: Symbol) -> Symbol {
        self.gensyms.push(symbol);
        Symbol::new(SymbolIndex::MAX_AS_U32 - self.gensyms.len() as u32 + 1)
    }

    fn is_gensymed(&mut self, symbol: Symbol) -> bool {
        symbol.0.as_usize() >= self.strings.len()
    }

    pub fn get(&self, symbol: Symbol) -> &str {
        match self.strings.get(symbol.0.as_usize()) {
            Some(string) => string,
            None => self.get(self.gensyms[(SymbolIndex::MAX_AS_U32 - symbol.0.as_u32()) as usize]),
        }
    }
}

// If an interner exists, return it. Otherwise, prepare a fresh one.
#[inline]
fn with_interner<T, F: FnOnce(&mut Interner) -> T>(f: F) -> T {
    let mut r = SYMBOL_TABLE
        .interner
        .write()
        .expect("unable to acquire write lock for symbol table");
    f(&mut *r)
    //GLOBALS.with(|globals| {
    //f(&mut *globals.symbol_interner.lock().expect("symbol interner lock was held"))
    //})
}

#[inline]
fn with_read_only_interner<T, F: FnOnce(&Interner) -> T>(f: F) -> T {
    let r = SYMBOL_TABLE
        .interner
        .read()
        .expect("unable to acquire read lock for symbol table");
    f(&*r)
}

/// Represents a string stored in the interner. Because the interner outlives any thread
/// which uses this type, we can safely treat `string` which points to interner data,
/// as an immortal string, as long as this type never crosses between threads.
#[derive(Clone, Copy, Hash, PartialOrd, Eq, Ord)]
pub struct LocalInternedString {
    string: &'static str,
    /// This type cannot be sent across threads, this emulates the
    /// behavior of !impl without the unsafe feature flag.
    dummy: PhantomData<*const u8>,
}

impl LocalInternedString {
    pub fn as_interned_str(self) -> InternedString {
        InternedString {
            symbol: Symbol::intern(self.string),
        }
    }

    pub fn get(&self) -> &'static str {
        self.string
    }
}

impl<U: ?Sized> ::std::convert::AsRef<U> for LocalInternedString
where
    str: ::std::convert::AsRef<U>,
{
    fn as_ref(&self) -> &U {
        self.string.as_ref()
    }
}

impl<T: ::std::ops::Deref<Target = str>> ::std::cmp::PartialEq<T> for LocalInternedString {
    fn eq(&self, other: &T) -> bool {
        self.string == other.deref()
    }
}

impl ::std::cmp::PartialEq<LocalInternedString> for str {
    fn eq(&self, other: &LocalInternedString) -> bool {
        self == other.string
    }
}

impl<'a> ::std::cmp::PartialEq<LocalInternedString> for &'a str {
    fn eq(&self, other: &LocalInternedString) -> bool {
        *self == other.string
    }
}

impl ::std::cmp::PartialEq<LocalInternedString> for String {
    fn eq(&self, other: &LocalInternedString) -> bool {
        self == other.string
    }
}

impl<'a> ::std::cmp::PartialEq<LocalInternedString> for &'a String {
    fn eq(&self, other: &LocalInternedString) -> bool {
        *self == other.string
    }
}

//impl !Send for LocalInternedString {}
//impl !Sync for LocalInternedString {}

impl ::std::ops::Deref for LocalInternedString {
    type Target = str;
    fn deref(&self) -> &str {
        self.string
    }
}

impl fmt::Debug for LocalInternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.string, f)
    }
}

impl fmt::Display for LocalInternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.string, f)
    }
}

/// Represents a string stored in the string interner.
#[derive(Clone, Copy, Eq)]
pub struct InternedString {
    symbol: Symbol,
}

impl InternedString {
    pub fn with<F: FnOnce(&str) -> R, R>(self, f: F) -> R {
        let str = with_interner(|interner| interner.get(self.symbol) as *const str);
        // This is safe because the interner keeps string alive until it is dropped.
        // We can access it because we know the interner is still alive since we use a
        // scoped thread local to access it, and it was alive at the beginning of this scope
        unsafe { f(&*str) }
    }

    pub fn as_symbol(self) -> Symbol {
        self.symbol
    }

    pub fn as_str(self) -> LocalInternedString {
        self.symbol.as_str()
    }
}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with(|str| str.hash(state))
    }
}

impl PartialOrd<InternedString> for InternedString {
    fn partial_cmp(&self, other: &InternedString) -> Option<Ordering> {
        if self.symbol == other.symbol {
            return Some(Ordering::Equal);
        }
        self.with(|self_str| other.with(|other_str| self_str.partial_cmp(other_str)))
    }
}

impl Ord for InternedString {
    fn cmp(&self, other: &InternedString) -> Ordering {
        if self.symbol == other.symbol {
            return Ordering::Equal;
        }
        self.with(|self_str| other.with(|other_str| self_str.cmp(&other_str)))
    }
}

impl<T: ::std::ops::Deref<Target = str>> PartialEq<T> for InternedString {
    fn eq(&self, other: &T) -> bool {
        self.with(|string| string == other.deref())
    }
}

impl PartialEq<InternedString> for InternedString {
    fn eq(&self, other: &InternedString) -> bool {
        self.symbol == other.symbol
    }
}

impl PartialEq<InternedString> for str {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| self == string)
    }
}

impl<'a> PartialEq<InternedString> for &'a str {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| *self == string)
    }
}

impl PartialEq<InternedString> for String {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| self == string)
    }
}

impl<'a> PartialEq<InternedString> for &'a String {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| *self == string)
    }
}

impl ::std::convert::From<InternedString> for String {
    fn from(val: InternedString) -> String {
        val.as_symbol().to_string()
    }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.with(|str| fmt::Debug::fmt(&str, f))
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.with(|str| fmt::Display::fmt(&str, f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interner_tests() {
        let mut i: Interner = Interner::default();
        // first one is zero:
        assert_eq!(i.intern("dog"), Symbol::new(0));
        // re-use gets the same entry:
        assert_eq!(i.intern("dog"), Symbol::new(0));
        // different string gets a different #:
        assert_eq!(i.intern("cat"), Symbol::new(1));
        assert_eq!(i.intern("cat"), Symbol::new(1));
        // dog is still at zero
        assert_eq!(i.intern("dog"), Symbol::new(0));
        assert_eq!(i.gensym("zebra"), Symbol::new(SymbolIndex::MAX_AS_U32));
        // gensym of same string gets new number:
        assert_eq!(i.gensym("zebra"), Symbol::new(SymbolIndex::MAX_AS_U32 - 1));
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym("dog"), Symbol::new(SymbolIndex::MAX_AS_U32 - 2));
    }

    #[test]
    fn interned_keywords_no_gaps() {
        let mut i = Interner::fresh();
        // Should already be interned with matching indexes
        for (sym, s) in symbols::DECLARED.iter() {
            assert_eq!(i.intern(&s), *sym)
        }
        // Should create a new symbol resulting in an index equal to the last entry in the table
        assert_eq!(i.intern("foo").as_u32(), (i.names.len() - 1) as u32);
    }

    #[test]
    fn unquote_string() {
        let i = Ident::from_str("\"after\"");
        assert_eq!(i.unquote_string().name, symbols::After);
    }

    #[test]
    fn unquote_atom() {
        let i = Ident::from_str("'after'");
        assert_eq!(i.unquote_atom().name, symbols::After);
    }
}
