use core::fmt::{self, Write};

use liblumen_binary::BinaryEntrySpecifier;
use liblumen_syntax_base::*;

use super::*;

pub struct PrettyPrinter<'b, 'a: 'b> {
    writer: &'b mut fmt::Formatter<'a>,
    indent: usize,
    trailing_newline: bool,
}
impl<'b, 'a: 'b> PrettyPrinter<'b, 'a> {
    pub fn new(writer: &'b mut fmt::Formatter<'a>) -> Self {
        Self {
            writer,
            indent: 0,
            trailing_newline: false,
        }
    }

    pub fn print_module(&mut self, module: &Module) -> fmt::Result {
        self.write_fmt(format_args!("module {} \n", module.name))?;
        self.write_str("    [\n")?;
        for (i, export) in module.exports.iter().enumerate() {
            if i > 0 {
                self.write_fmt(format_args!(
                    ",\n      {}/{}",
                    export.function, export.arity
                ))?;
            } else {
                self.write_fmt(format_args!("      {}/{}", export.function, export.arity))?;
            }
        }
        self.write_str("\n    ]")?;

        for function in module.functions.iter() {
            self.write_str("\n\n")?;
            self.indent += 1;
            self.print_function(function)?;
            self.indent -= 1;
        }

        self.write_str("\nend")
    }

    pub fn print_function(&mut self, function: &Function) -> fmt::Result {
        self.indent()?;
        self.write_fmt(format_args!("{}", &function.name))?;
        self.write_vars(function.vars.as_slice(), '(', ')')?;
        self.write_str(" {")?;
        self.indent += 1;
        self.nl()?;
        self.indent()?;
        self.print_body(function.body.as_ref())?;
        self.indent -= 1;
        self.nl()?;
        self.indent()?;
        self.write_str("}")
    }

    fn print_body(&mut self, expr: &Expr) -> fmt::Result {
        match expr {
            // Block-like expressions can be nested and require indentation
            Expr::Alt(_)
            | Expr::Catch(_)
            | Expr::Fun(_)
            | Expr::Guard(_)
            | Expr::If(_)
            | Expr::LetRec(_)
            | Expr::LetRecGoto(_)
            | Expr::Match(_)
            | Expr::Select(_)
            | Expr::Seq(_)
            | Expr::Set(_)
            | Expr::Try(_)
            | Expr::TryEnter(_) => self.print_block_like(expr),
            // Value-like expressions are used as arguments to instructions
            Expr::Values(_)
            | Expr::Literal(_)
            | Expr::Var(_)
            | Expr::Local(_)
            | Expr::Remote(_) => self.print_value(expr),
            // Instructions are value-producing operations
            Expr::Binary(_)
            | Expr::BinaryInt(_)
            | Expr::BinarySegment(_)
            | Expr::BinaryEnd(_)
            | Expr::Cons(_)
            | Expr::Tuple(_)
            | Expr::Map(_)
            | Expr::Alias(_)
            | Expr::Bif(_)
            | Expr::Break(_)
            | Expr::Call(_)
            | Expr::Enter(_)
            | Expr::Goto(_)
            | Expr::Put(_)
            | Expr::Return(_)
            | Expr::Test(_) => self.print_expr(expr),
        }
    }

    fn print_block_like(&mut self, expr: &Expr) -> fmt::Result {
        match expr {
            Expr::Alt(Alt { first, then, .. }) => {
                self.write_str("alt")?;
                self.nl()?;
                self.indent()?;
                self.write_str("do ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(first.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("else ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(then.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            Expr::Catch(Catch { body, ret, .. }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("catch ")?;
                self.indent += 1;
                self.print_body(body.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            Expr::Fun(fun) => self.print_fun(&fun),
            Expr::Guard(Guard { clauses, .. }) => {
                self.write_str("guard")?;
                for clause in clauses.iter() {
                    self.nl()?;
                    self.indent()?;
                    self.write_str("| ")?;
                    self.indent += 1;
                    self.print_body(clause.guard.as_ref())?;
                    self.write_str(" => ")?;
                    self.nl()?;
                    self.indent()?;
                    self.print_body(clause.body.as_ref())?;
                    self.indent -= 1;
                }
                Ok(())
            }
            Expr::If(If {
                cond,
                then_body,
                else_body,
                ret,
                ..
            }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("if ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(cond.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("then ")?;
                self.indent += 1;
                self.print_body(then_body.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("else ")?;
                self.indent += 1;
                self.print_body(else_body.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            Expr::LetRec(ILetRec { defs, .. }) => {
                self.write_str("letrec")?;
                self.indent += 1;
                for (name, fun) in defs.iter() {
                    self.nl()?;
                    self.indent()?;
                    self.write_var(&name)?;
                    self.write_str(" = ")?;
                    self.print_fun(fun)?;
                }
                self.indent -= 1;
                Ok(())
            }
            Expr::LetRecGoto(LetRecGoto {
                label,
                vars,
                first,
                then,
                ret,
                ..
            }) => {
                self.write_ret(ret.as_slice())?;
                self.write_fmt(format_args!("letrec.goto {}", label))?;
                self.write_vars(vars.as_slice(), '(', ')')?;
                self.nl()?;
                self.indent()?;
                self.write_str("do ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(first.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("then ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(then.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            Expr::Match(Match { body, ret, .. }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("match ")?;
                self.nl()?;
                self.indent += 1;
                self.indent()?;
                self.print_body(body.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            Expr::Select(Select { var, types, .. }) => {
                self.write_str("select ")?;
                self.write_var(&var)?;
                for clause in types.iter() {
                    for value in clause.values.iter() {
                        self.nl()?;
                        self.indent()?;
                        self.write_str("|")?;
                        self.write_match_type(clause.ty)?;
                        self.print_value(value.value.as_ref())?;
                        self.write_str(" => ")?;
                        self.indent += 1;
                        self.nl()?;
                        self.indent()?;
                        self.print_body(value.body.as_ref())?;
                        self.indent -= 1;
                    }
                }
                Ok(())
            }
            Expr::Seq(Seq { arg, body, .. }) => {
                self.print_body(arg.as_ref())?;
                self.nl()?;
                self.indent()?;
                self.print_body(body.as_ref())?;
                Ok(())
            }
            Expr::Set(ISet {
                vars, arg, body, ..
            }) => {
                self.write_str("set ")?;
                self.write_vars(vars.as_slice(), '<', '>')?;
                self.write_str(" = ")?;
                self.print_body(arg.as_ref())?;
                if let Some(body) = body.as_ref() {
                    self.nl()?;
                    self.indent()?;
                    self.write_str("in ")?;
                    self.indent += 1;
                    self.print_body(body.as_ref())?;
                    self.indent -= 1;
                }
                Ok(())
            }
            Expr::Try(Try {
                arg,
                vars,
                body,
                evars,
                handler,
                ret,
                ..
            }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("try ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(arg.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("then ")?;
                self.write_vars(vars.as_slice(), '(', ')')?;
                self.write_str(" -> ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(body.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("catch ")?;
                self.write_vars(evars.as_slice(), '(', ')')?;
                self.write_str(" -> ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(handler.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            Expr::TryEnter(TryEnter {
                arg,
                vars,
                body,
                evars,
                handler,
                ..
            }) => {
                self.write_str("try.enter ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(arg.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("then ")?;
                self.write_vars(vars.as_slice(), '(', ')')?;
                self.write_str(" -> ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(body.as_ref())?;
                self.indent -= 1;
                self.nl()?;
                self.indent()?;
                self.write_str("catch ")?;
                self.write_vars(evars.as_slice(), '(', ')')?;
                self.write_str(" -> ")?;
                self.indent += 1;
                self.nl()?;
                self.indent()?;
                self.print_body(handler.as_ref())?;
                self.indent -= 1;
                Ok(())
            }
            _ => unimplemented!(),
        }
    }

    fn print_fun(&mut self, fun: &IFun) -> fmt::Result {
        self.write_str("fun ")?;
        self.write_vars(fun.vars.as_slice(), '(', ')')?;
        self.write_str(" { ")?;
        self.indent += 1;
        self.nl()?;
        self.indent()?;
        self.print_body(fun.body.as_ref())?;
        self.indent -= 1;
        self.nl()?;
        self.indent()?;
        self.write_char('}')
    }

    fn print_expr(&mut self, expr: &Expr) -> fmt::Result {
        match expr {
            Expr::Binary(Binary { segment, .. }) => {
                self.write_str("binary (")?;
                let mut first = true;
                let mut next = segment.as_ref();
                loop {
                    match next {
                        Expr::BinaryEnd(_) => break,
                        Expr::BinaryInt(BinarySegment {
                            size,
                            value,
                            spec,
                            next: nxt,
                            ..
                        }) => {
                            if !first {
                                self.write_str(", ")?;
                            } else {
                                first = false;
                            }
                            next = nxt.as_ref();
                            self.write_str("binary.int(")?;
                            self.print_value(value.as_ref())?;
                            if let Some(sz) = size.as_deref() {
                                self.write_str(":")?;
                                self.print_value(sz)?;
                            }
                            self.write_spec(*spec, true)?;
                            self.write_char(')')?;
                        }
                        Expr::BinarySegment(BinarySegment {
                            size,
                            value,
                            spec,
                            next: nxt,
                            ..
                        }) => {
                            if !first {
                                self.write_str(", ")?;
                            } else {
                                first = false;
                            }
                            next = nxt.as_ref();
                            self.write_str("binary.seg(")?;
                            self.print_value(value.as_ref())?;
                            if let Some(sz) = size.as_deref() {
                                self.write_str(":")?;
                                self.print_value(sz)?;
                            }
                            self.write_spec(*spec, false)?;
                            self.write_char(')')?;
                        }
                        Expr::Var(v) => {
                            self.write_var(v)?;
                            break;
                        }
                        other => panic!("unexpected binary next instruction: {:#?}", other),
                    }
                }
                self.write_char(')')
            }
            Expr::BinaryInt(BinarySegment {
                size,
                value,
                spec,
                next,
                ..
            }) => {
                self.write_str("binary.int (")?;
                self.print_value(value.as_ref())?;
                if let Some(sz) = size.as_deref() {
                    self.write_str(":")?;
                    self.print_value(sz)?;
                }
                self.write_spec(*spec, true)?;
                self.write_str(")")?;
                self.nl()?;
                self.indent()?;
                self.print_expr(next.as_ref())
            }
            Expr::BinarySegment(BinarySegment {
                size,
                value,
                spec,
                next,
                ..
            }) => {
                self.write_str("binary.seg (")?;
                self.print_value(value.as_ref())?;
                if let Some(sz) = size.as_deref() {
                    self.write_str(":")?;
                    self.print_value(sz)?;
                }
                self.write_spec(*spec, false)?;
                self.write_str(")")?;
                self.nl()?;
                self.indent()?;
                self.print_expr(next.as_ref())
            }
            Expr::BinaryEnd(_) => self.write_str("binary.end"),
            Expr::Cons(Cons { head, tail, .. }) => {
                self.write_str("cons ")?;
                self.print_value(head.as_ref())?;
                self.write_str(", ")?;
                self.print_value(tail.as_ref())
            }
            Expr::Tuple(Tuple { elements, .. }) => {
                self.write_str("tuple ")?;
                for (i, e) in elements.iter().enumerate() {
                    if i > 0 {
                        self.write_str(", ")?;
                    }
                    self.print_value(e)?;
                }
                Ok(())
            }
            Expr::Map(Map { var, pairs, .. }) => {
                self.write_str("map ")?;
                self.print_value(var.as_ref())?;
                if !pairs.is_empty() {
                    self.write_str(", ")?;
                    for pair in pairs.iter() {
                        self.print_value(pair.key.as_ref())?;
                        self.write_str(" := ")?;
                        self.print_value(pair.value.as_ref())?;
                    }
                }
                Ok(())
            }
            Expr::Alias(IAlias { vars, pattern, .. }) => {
                self.write_str("alias ")?;
                self.write_vars(vars.as_slice(), '<', '>')?;
                self.write_str(" = ")?;
                self.print_expr(pattern.as_ref())
            }
            Expr::Bif(Bif { op, args, ret, .. }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("bif ")?;
                self.write_fmt(format_args!("{}", op))?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Break(Break { args, .. }) => {
                self.write_str("break ")?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Call(Call {
                callee, args, ret, ..
            }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("call ")?;
                self.print_value(callee.as_ref())?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Enter(Enter { callee, args, .. }) => {
                self.write_str("enter ")?;
                self.print_value(callee.as_ref())?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Goto(Goto { label, args, .. }) => {
                self.write_fmt(format_args!("goto {}", label))?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Put(Put { arg, ret, .. }) => {
                self.write_ret(ret.as_slice())?;
                self.write_str("put ")?;
                self.print_value(arg.as_ref())
            }
            Expr::Return(Return { args, .. }) => {
                self.write_str("return ")?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Test(Test { op, args, .. }) => {
                self.write_fmt(format_args!("test {}", op))?;
                self.write_values(args.as_slice(), '(', ')')
            }
            Expr::Var(v) => self.write_var(v),
            other => unimplemented!("printing not implemented for {:#?}", other),
        }
    }

    fn print_value(&mut self, expr: &Expr) -> fmt::Result {
        match expr {
            Expr::Literal(value) => printing::print_literal(self.writer, value),
            Expr::Values(IValues { values, .. }) => {
                self.write_char('<')?;
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        self.write_str(", ")?;
                    }
                    self.print_value(value)?;
                }
                return self.write_char('>');
            }
            Expr::Var(v) => self.write_var(v),
            Expr::Local(name) => self.write_fmt(format_args!("local {}", name)),
            Expr::Remote(Remote::Static(name)) => self.write_fmt(format_args!("remote {}", name)),
            Expr::Remote(Remote::Dynamic(m, f)) => {
                self.write_str("remote ")?;
                self.print_value(m.as_ref())?;
                self.write_char(':')?;
                self.print_value(f.as_ref())
            }
            other => self.print_expr(other),
        }
    }

    fn write_values(&mut self, values: &[Expr], l: char, r: char) -> fmt::Result {
        self.write_char(l)?;
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                self.write_str(", ")?;
            }
            self.print_value(v)?;
        }
        self.write_char(r)
    }

    fn write_vars(&mut self, vars: &[Var], l: char, r: char) -> fmt::Result {
        self.write_char(l)?;
        for (i, v) in vars.iter().enumerate() {
            if i > 0 {
                self.write_str(", ")?;
            }
            self.write_var(v)?;
        }
        self.write_char(r)
    }

    fn write_var(&mut self, var: &Var) -> fmt::Result {
        match var {
            Var {
                name, arity: None, ..
            } => self.write_fmt(format_args!("{}", name.name)),
            Var {
                name,
                arity: Some(arity),
                ..
            } => self.write_fmt(format_args!("{}/{}", name.name, arity)),
        }
    }

    fn write_ret(&mut self, ret: &[Expr]) -> fmt::Result {
        if ret.is_empty() {
            return Ok(());
        }
        if ret.len() > 1 {
            self.write_values(ret, '<', '>')?;
        } else {
            self.print_value(&ret[0])?;
        }
        self.write_str(" = ")
    }

    fn write_match_type(&mut self, ty: MatchType) -> fmt::Result {
        match ty {
            MatchType::Binary
            | MatchType::BinaryInt
            | MatchType::BinarySegment
            | MatchType::BinaryEnd
            | MatchType::Cons
            | MatchType::Tuple
            | MatchType::Map
            | MatchType::Atom
            | MatchType::Nil
            | MatchType::Literal => self.write_char(' '),
            MatchType::Float => self.write_str(" float "),
            MatchType::Int => self.write_str(" integer "),
            MatchType::Var => self.write_str(" var "),
        }
    }

    fn write_spec(&mut self, spec: BinaryEntrySpecifier, is_int: bool) -> fmt::Result {
        use liblumen_binary::Endianness;

        self.write_char('/')?;
        match spec {
            BinaryEntrySpecifier::Integer {
                signed,
                endianness,
                unit,
            } => {
                if !is_int {
                    self.write_str("integer-")?;
                }
                if signed {
                    self.write_str("signed")?;
                } else {
                    self.write_str("unsigned")?;
                }
                self.write_fmt(format_args!("-{}-unit({})", endianness, unit))
            }
            BinaryEntrySpecifier::Float { endianness, unit } => {
                self.write_fmt(format_args!("float-{}-unit({})", endianness, unit))
            }
            BinaryEntrySpecifier::Binary { unit } if unit == 8 => self.write_str("binary"),
            BinaryEntrySpecifier::Binary { unit } if unit == 1 => self.write_str("bitstring"),
            BinaryEntrySpecifier::Binary { unit } => {
                self.write_fmt(format_args!("bitstring-unit({})", unit))
            }
            BinaryEntrySpecifier::Utf8 => self.write_str("utf8"),
            BinaryEntrySpecifier::Utf16 { endianness } if endianness == Endianness::Big => {
                self.write_str("utf16")
            }
            BinaryEntrySpecifier::Utf16 { endianness } => {
                self.write_fmt(format_args!("utf16-{}", endianness))
            }
            BinaryEntrySpecifier::Utf32 { endianness } if endianness == Endianness::Big => {
                self.write_str("utf32")
            }
            BinaryEntrySpecifier::Utf32 { endianness } => {
                self.write_fmt(format_args!("utf32-{}", endianness))
            }
        }
    }

    #[inline]
    fn write_fmt(&mut self, f: fmt::Arguments) -> fmt::Result {
        let s = format!("{}", f);
        self.write_str(&s)
    }

    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.trailing_newline = s.ends_with('\n');
        self.writer.write_str(s)
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.trailing_newline = c == '\n';
        self.writer.write_char(c)
    }

    fn indent(&mut self) -> fmt::Result {
        if self.indent == 0 {
            return Ok(());
        }
        // Only indent if there is a trailing newline
        if self.trailing_newline {
            self.trailing_newline = false;
            for _ in 0..self.indent {
                self.writer.write_str("  ")?;
            }
        }
        Ok(())
    }

    #[inline]
    fn nl(&mut self) -> fmt::Result {
        // Only add a newline if there isn't a trailing newline
        if !self.trailing_newline {
            self.trailing_newline = true;
            self.writer.write_char('\n')
        } else {
            Ok(())
        }
    }
}
