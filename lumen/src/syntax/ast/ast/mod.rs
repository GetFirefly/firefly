macro_rules! impl_from {
    ($to:ident :: $constructor:ident ($from:ty)) => {
        impl ::std::convert::From<$from> for $to {
            fn from(x: $from) -> Self {
                $to::$constructor(::std::convert::From::from(x))
            }
        }
    };
}

macro_rules! impl_node {
    ($x:ident <$a:ident, $b:ident>) => {
        impl<$a, $b> crate::syntax::ast::ast::Node for $x<$a, $b> {
            fn line(&self) -> crate::syntax::ast::ast::LineNum {
                self.line
            }
        }
    };
    ($x:ident <$a:ident>) => {
        impl<$a> crate::syntax::ast::ast::Node for $x<$a> {
            fn line(&self) -> crate::syntax::ast::ast::LineNum {
                self.line
            }
        }
    };
    ($x:ty) => {
        impl crate::syntax::ast::ast::Node for $x {
            fn line(&self) -> crate::syntax::ast::ast::LineNum {
                self.line
            }
        }
    };
}

use crate::syntax;

pub mod clause;
pub mod common;
pub mod expr;
pub mod form;
pub mod guard;
pub mod literal;
pub mod pat;
pub mod ty;

pub type LineNum = i32;
pub type Arity = u32;

pub trait Node {
    fn line(&self) -> LineNum;
}

#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub forms: Vec<self::form::Form>,
}
impl std::convert::From<syntax::parser::cst::ModuleDecl> for ModuleDecl {
    fn from(_parsed: syntax::parser::cst::ModuleDecl) -> Self {
        unimplemented!()
    }
}

// impl std::convert::From<crate::syntax::parser::cst::ModuleDecl> for ModuleDecl {
//     fn from(parsed: crate::syntax::parser::cst::ModuleDecl) -> Self {
//         use crate::syntax::parser::cst;
//         use crate::syntax::parser::forms;

//         let mut forms: Vec<self::form::Form> = Vec::new();
//         for form in parsed.forms.iter() {
//             match form {
//                 cst::ModuleAttr(forms::ModuleAttr { module_name, .. }) =>
//                     form::Form::Module(form::ModuleAttr { name })
//                 cst::ExportAttr(forms::ExportAttr { exports, .. }) => {
//                     let funs = exports.iter()
//                                       .map(|e| form::Export { name: e.name.value().unwrap(), arity: e.arity.to_u32() })
//                                       .collect();
//                     form::Form::Export(form::ExportAttr { funs })
//                 },
//                 cst::ExportTypeAttr(forms::ExportTypeAttr { exports, .. }) => {
//                     let types = exports.iter()
//                                       .map(|e| form::ExportType { typ: e.name.value().unwrap(), arity: e.arity.to_u32() })
//                                       .collect();
//                     form::Form::ExportType(form::ExportTypeAttr { types, .. })
//                 },
//                 cst::ImportAttr(forms::ImportAttr { module_name, imports, .. }) => {
//                     let imports = imports.iter()
//                                          .map(|e| form::ImportAttr { name: module_name, funs: imports })
//                                          .collect();
//                     form::Form::Import(form::ImportAttr),
//                 },
//                 cst::FileAttr(forms::FileAttr { file_name, line_num, .. }) =>
//                     form::Form::File(form::FileAttr { original_file: file_name, original_line: line_num }),
//                 cst::WildAttr(forms::WildAttr { attr_name, attr_value, .. }) => {
//                     let value = match attr_value {
//                         LexicalToken::Atom(ref t) => etf::Term::Atom { name: t.value() },
//                         LexicalToken::Char(ref t) => etf::Term::FixedInteger { value: t.value() },
//                         LexicalToken::Float(ref t) => etf::Term::Float { value: t.value() },
//                         LexicalToken::Integer(ref t) => etf::Term::BigInteger { value: t.value() },
//                         LexicalToken::String(ref t) => etf::Term::Binary { value: t.value() },
//                         _ => unimplemented!(),
//                     };
//                     form::Form::Attr(form::WildAttr { name: attr_name, value })
//                 },
//                 cst::FunSpec(forms::FunSpec) =>
//                     // TODO
//                     form::Form::Spec(form::Spec),
//                 cst::CallbackSpec(forms::CallbackSpec) =>
//                     form::Form::Behaviour(forms::Behaviour),
//                 cst::FunDecl(forms::FunDecl) =>
//                     form::Form::Fun(forms::Fun),
//                 cst::RecordDecl(forms::RecordDecl) =>
//                     form::Form::Record(forms::Record),
//                 cst::TypeDecl(forms::TypeDecl) =>
//                     form::Form::Type(forms::Type),
//             }
//         }
//         forms.push(form::Form::Eof(forms::Eof { .. }))
//     }
// }
