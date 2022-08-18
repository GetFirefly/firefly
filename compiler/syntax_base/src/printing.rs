use std::fmt::{self, Write};

use liblumen_binary::Bitstring;
use liblumen_intern::Symbol;

use crate::{Annotation, Lit, Literal};

pub fn print_literal(f: &mut fmt::Formatter, literal: &Literal) -> fmt::Result {
    print_lit(f, &literal.value)
}

pub fn print_lit(f: &mut fmt::Formatter, lit: &Lit) -> fmt::Result {
    match lit {
        Lit::Atom(a) => write!(f, "{}", a),
        Lit::Integer(i) => write!(f, "{}", i),
        Lit::Float(n) => write!(f, "{}", n),
        Lit::Nil => f.write_str("[]"),
        Lit::Cons(ref h, ref t) => {
            f.write_char('[')?;
            print_literal(f, h)?;
            f.write_str(" | ")?;
            print_literal(f, t)?;
            f.write_char(']')
        }
        Lit::Tuple(es) => {
            f.write_char('{')?;
            for (i, e) in es.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                print_literal(f, e)?;
            }
            f.write_char('}')
        }
        Lit::Map(map) => {
            f.write_str("#{")?;
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                print_literal(f, k)?;
                f.write_str(" := ")?;
                print_literal(f, v)?;
            }
            f.write_char('}')
        }
        Lit::Binary(bitvec) => write!(f, "{}", bitvec.display()),
    }
}

pub fn print_annotation(f: &mut fmt::Formatter, sym: &Symbol, value: &Annotation) -> fmt::Result {
    match value {
        Annotation::Unit => write!(f, "{}", sym),
        Annotation::Term(ref value) => {
            write!(f, "{{{}, ", sym)?;
            print_literal(f, value)?;
            f.write_str("}")
        }
        Annotation::Vars(vars) => {
            write!(f, "{{{}, [", sym)?;
            for (i, id) in vars.iter().enumerate() {
                if i > 0 {
                    write!(f, ",{}", id)?;
                } else {
                    write!(f, "{}", id)?;
                }
            }
            write!(f, "]}}")
        }
        Annotation::Type(ty) => write!(f, "{{type, {}}}", &ty),
    }
}
