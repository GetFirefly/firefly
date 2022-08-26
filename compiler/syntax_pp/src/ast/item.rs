use firefly_diagnostics::{Spanned, SourceSpan};
use firefly_intern::Ident;
use firefly_number::{Integer, Float};

#[derive(Debug)]
pub struct Root {
    pub items: Vec<Item>,
}
impl Root {
    pub fn span(&self) -> SourceSpan {
        self.items
            .first()
            .map(|i| i.span())
            .unwrap_or(SourceSpan::UNKNOWN)
    }
}

#[derive(Debug)]
pub enum Item {
    Lit(Literal),
    Attribute(Spanned<Attr>),
    Function(Spanned<Func>),
    Tuple(Spanned<Tuple>),
    List(Spanned<List>),
    Eof(SourceSpan),
}

impl Item {
    pub fn tuple(&self) -> Option<&Tuple> {
        match self {
            Item::Tuple(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn atom(&self) -> Option<Ident> {
        match self {
            Item::Atom(inner) => Some(*inner),
            _ => None,
        }
    }

    pub fn string(&self) -> Option<Ident> {
        match self {
            Item::String(inner) => Some(*inner),
            _ => None,
        }
    }

    pub fn integer(&self) -> Option<&Int> {
        match self {
            Item::Int(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn float(&self) -> Option<&Float> {
        match self {
            Item::Float(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn list(&self) -> Option<&List> {
        match self {
            Item::List(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn list_iter(&self) -> Option<impl Iterator<Item = &Item>> {
        self.list().map(|v| ListIterator {
            curr: Some(v),
            idx: 0,
        })
    }

    pub fn span(&self) -> SourceSpan {
        match self {
            Item::Atom(ident) => ident.span,
            Item::String(ident) => ident.span,
            Item::Int(int) => int.span,
            Item::Tuple(tup) => tup.span,
            Item::List(list) => list.span,
            Item::Float(float) => float.span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Literal {
    Atom(Ident),
    Bool(Ident),
    Int(SourceSpan, Integer),
    Float(SourceSpan, Float),
    String(Ident),
}

#[derive(Debug)]
pub struct Tuple(Vec<Item>);

#[derive(Debug)]
pub struct List {
    pub span: SourceSpan,
    pub head: Vec<Item>,
    pub tail: Option<Box<Item>>,
}

pub struct ListIterator<'a> {
    curr: Option<&'a List>,
    idx: usize,
}
impl<'a> Iterator for ListIterator<'a> {
    type Item = &'a Item;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(list) = self.curr {
            if let Some(result) = list.heads.get(self.idx) {
                self.idx += 1;
                return Some(result);
            } else {
                self.idx = 0;
                if let Some(tail) = &list.tail {
                    if let Some(tail_list) = tail.list() {
                        self.curr = Some(tail_list);
                        return self.next();
                    } else {
                        unimplemented!()
                    }
                } else {
                    self.curr = None;
                    return None;
                }
            }
        } else {
            None
        }
    }
}
