use std::fmt::{self, Write};

use liblumen_binary::{BinaryEntrySpecifier, Endianness};
use liblumen_syntax_base::*;

use crate::*;

pub struct PrettyPrinter<'b, 'a: 'b> {
    writer: &'b mut fmt::Formatter<'a>,
    indent: usize,
}
impl<'b, 'a: 'b> PrettyPrinter<'b, 'a> {
    pub fn new(writer: &'b mut fmt::Formatter<'a>) -> Self {
        Self { writer, indent: 0 }
    }

    pub fn print_module(&mut self, module: &Module) -> fmt::Result {
        write!(self.writer, "module {}\n", module.name)?;
        write!(self.writer, "  [\n")?;
        for (i, export) in module.exports.iter().enumerate() {
            if i > 0 {
                write!(self.writer, ",\n    {}/{}", export.function, export.arity)?;
            } else {
                write!(self.writer, "    {}/{}", export.function, export.arity)?;
            }
        }
        write!(self.writer, "\n  ]\n\n")?;

        for (_, function) in module.functions.iter() {
            write!(
                self.writer,
                "{}/{} =\n",
                function.fun.name,
                function.fun.vars.len()
            )?;
            self.indent += 2;
            self.print_fun(&function.fun)?;
            self.indent -= 2;
        }

        write!(self.writer, "\n\nend")
    }

    pub fn print_fun(&mut self, fun: &Fun) -> fmt::Result {
        self.writer.write_str("fun (")?;
        for (i, v) in fun.vars.iter().enumerate() {
            if i > 0 {
                write!(self.writer, ", {}", v.name)?;
            } else {
                write!(self.writer, "{}", v.name)?;
            }
        }
        self.writer.write_str(") ->\n")?;
        self.indent += 2;
        self.indent()?;
        self.print_expr(fun.body.as_ref())?;
        self.indent -= 2;
        self.writer.write_str("end\n")?;
        self.print_annotations(&fun.annotations)
    }

    pub fn print_ifun(&mut self, fun: &IFun) -> fmt::Result {
        self.writer.write_str("ifun (")?;
        for (i, v) in fun.vars.iter().enumerate() {
            if i > 0 {
                write!(self.writer, ", {}", v.name)?;
            } else {
                write!(self.writer, "{}", v.name)?;
            }
        }
        self.writer.write_str(") ->")?;
        for clause in fun.clauses.iter() {
            self.indent += 2;
            self.writer.write_char('\n')?;
            self.print_iclause(clause)?;
            self.indent -= 2;
        }
        self.writer.write_str("end\n")?;
        self.print_annotations(&fun.annotations)
    }

    pub fn print_args(&mut self, exprs: &[Expr]) -> fmt::Result {
        if exprs.len() > 1 {
            self.print_exprs(exprs, true, true)
        } else {
            self.writer.write_char('(')?;
            self.print_exprs(exprs, true, false)?;
            self.writer.write_char(')')
        }
    }

    pub fn print_iargs(&mut self, exprs: &[IExpr]) -> fmt::Result {
        if exprs.len() > 1 {
            self.print_iexprs(exprs, true, true)
        } else {
            self.writer.write_char('(')?;
            self.print_iexprs(exprs, true, false)?;
            self.writer.write_char(')')
        }
    }

    pub fn print_exprs(&mut self, exprs: &[Expr], inline: bool, parenthesize: bool) -> fmt::Result {
        self.indent += 2;
        if !inline {
            self.writer.write_char('\n')?;
        }
        let parenthesize = parenthesize && (inline && exprs.len() > 1);
        if parenthesize {
            self.writer.write_char('(')?;
        }
        for (i, expr) in exprs.iter().enumerate() {
            if i > 0 {
                if inline {
                    self.writer.write_str(", ")?;
                } else {
                    self.writer.write_str(",\n")?;
                    self.indent()?;
                }
            } else {
                if !inline {
                    self.indent()?;
                }
            }
            self.print_expr(expr)?;
        }
        if parenthesize {
            self.writer.write_char(')')?;
        }
        self.indent -= 2;
        if !inline {
            self.writer.write_char('\n')?;
        }
        Ok(())
    }

    pub fn print_iexprs(
        &mut self,
        exprs: &[IExpr],
        inline: bool,
        parenthesize: bool,
    ) -> fmt::Result {
        self.indent += 2;
        if !inline {
            self.writer.write_char('\n')?;
        }
        let parenthesize = parenthesize && (inline && exprs.len() > 1);
        if parenthesize {
            self.writer.write_char('(')?;
        }
        for (i, expr) in exprs.iter().enumerate() {
            if i > 0 {
                if inline {
                    self.writer.write_str(", ")?;
                } else {
                    self.writer.write_str(",\n")?;
                    self.indent()?;
                }
            } else {
                if !inline {
                    self.indent()?;
                }
            }
            self.print_iexpr(expr)?;
        }
        if parenthesize {
            self.writer.write_char(')')?;
        }
        self.indent -= 2;
        if !inline {
            self.writer.write_char('\n')?;
        }
        Ok(())
    }

    pub fn print_expr(&mut self, expr: &Expr) -> fmt::Result {
        match expr {
            Expr::Alias(expr) => {
                write!(self.writer, "{} = ", expr.var.name())?;
                self.print_expr(expr.pattern.as_ref())
            }
            Expr::Apply(apply) => {
                self.writer.write_str("apply ")?;
                self.print_expr(apply.callee.as_ref())?;
                write!(self.writer, "/{}", apply.args.len())?;
                self.print_args(apply.args.as_slice())
            }
            Expr::Binary(bin) => {
                self.writer.write_str("<<")?;
                for (i, segment) in bin.segments.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    self.print_bitstring(segment)?;
                }
                self.writer.write_str(">>")
            }
            Expr::Call(call) => {
                self.writer.write_str("call ")?;
                self.print_expr(call.module.as_ref())?;
                self.writer.write_char(':')?;
                self.print_expr(call.function.as_ref())?;
                write!(self.writer, "/{}", call.args.len())?;
                self.print_args(call.args.as_slice())
            }
            Expr::Case(expr) => {
                self.writer.write_str("case ")?;
                self.print_expr(expr.arg.as_ref())?;
                self.writer.write_str(" of\n")?;
                self.indent += 2;
                for (i, clause) in expr.clauses.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char('\n')?;
                    }
                    self.print_clause(clause)?;
                }
                self.indent -= 2;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            Expr::Catch(expr) => {
                self.writer.write_str("catch ")?;
                self.print_expr(expr.body.as_ref())?;
                self.writer.write_str(" end")
            }
            Expr::Cons(cons) => {
                self.writer.write_str("[")?;
                self.print_expr(cons.head.as_ref())?;
                self.writer.write_str(" | ")?;
                self.print_expr(cons.tail.as_ref())?;
                self.writer.write_str("]")
            }
            Expr::Fun(ref fun) => {
                self.indent += 2;
                self.print_fun(fun)?;
                self.indent -= 2;
                Ok(())
            }
            Expr::If(expr) => {
                self.writer.write_str("if ")?;
                self.print_expr(expr.guard.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent += 2;
                self.indent()?;
                self.writer.write_str("then ")?;
                self.indent += 2;
                self.print_expr(expr.then_body.as_ref())?;
                self.indent -= 2;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("else ")?;
                self.indent += 2;
                self.print_expr(expr.else_body.as_ref())?;
                self.indent -= 4;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            Expr::Let(expr) => {
                self.writer.write_str("let <")?;
                for (i, v) in expr.vars.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    write!(self.writer, "{}", v.name())?;
                }
                self.writer.write_str("> = ")?;
                self.indent += 2;
                self.print_expr(expr.arg.as_ref())?;
                self.indent -= 2;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("in ")?;
                self.writer.write_char('\n')?;
                self.indent += 2;
                self.indent()?;
                self.print_expr(expr.body.as_ref())?;
                self.indent -= 2;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            Expr::LetRec(expr) => {
                self.writer.write_str("letrec")?;
                self.indent += 2;
                for (i, (binding, expr)) in expr.defs.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(",\n")?;
                    } else {
                        self.writer.write_char('\n')?;
                    }
                    self.indent()?;
                    write!(self.writer, "{} = ", binding.name())?;
                    self.indent += 2;
                    self.print_expr(expr)?;
                    self.indent -= 2;
                }
                self.writer.write_char('\n')?;
                self.indent -= 2;
                self.indent()?;
                self.writer.write_str("in ")?;
                self.writer.write_char('\n')?;
                self.indent += 2;
                self.indent()?;
                self.print_expr(expr.body.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent -= 2;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            Expr::Literal(ref literal) => printing::print_literal(self.writer, literal),
            Expr::Map(map) => {
                let print_arg = match map.arg.as_ref() {
                    Expr::Literal(Literal {
                        value: Lit::Map(m), ..
                    }) if m.is_empty() => false,
                    _ => true,
                };
                if print_arg {
                    self.print_expr(map.arg.as_ref())?;
                }
                self.writer.write_str("#{")?;
                for (i, pair) in map.pairs.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    self.indent += 2;
                    self.print_expr(pair.key.as_ref())?;
                    match pair.op {
                        MapOp::Exact => self.writer.write_str(" := ")?,
                        _ => self.writer.write_str(" => ")?,
                    }
                    self.print_expr(pair.value.as_ref())?;
                    self.indent -= 2;
                }
                self.writer.write_char('}')
            }
            Expr::PrimOp(op) => {
                write!(self.writer, "primop {}", op.name)?;
                self.print_args(op.args.as_slice())
            }
            Expr::Receive(recv) => {
                self.writer.write_str("receive\n")?;
                for (i, clause) in recv.clauses.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char('\n')?;
                    }
                    self.print_clause(clause)?;
                }
                if !recv.clauses.is_empty() {
                    self.writer.write_char('\n')?;
                }
                self.indent()?;
                self.writer.write_str("after\n")?;
                self.indent += 2;
                self.indent()?;
                self.print_expr(recv.timeout.as_ref())?;
                self.writer.write_str(" ->\n")?;
                self.indent += 2;
                self.indent()?;
                self.print_expr(recv.action.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent -= 4;
                self.indent()?;
                self.writer.write_str("end")
            }
            Expr::Seq(expr) => {
                self.print_expr(expr.arg.as_ref())?;
                self.writer.write_str(", ")?;
                self.print_expr(expr.body.as_ref())
            }
            Expr::Try(expr) => {
                self.writer.write_str("try ")?;
                self.print_expr(expr.arg.as_ref())?;
                self.writer.write_str(" of\n")?;
                self.indent += 2;
                self.indent()?;
                self.writer.write_char('<')?;
                for (i, v) in expr.vars.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    write!(self.writer, "{}", v.name())?;
                }
                self.writer.write_str("> ->\n")?;
                self.indent += 2;
                self.indent()?;
                self.print_expr(expr.body.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent -= 2;
                self.indent()?;
                self.writer.write_str("catch ")?;
                for (i, evar) in expr.evars.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char(':')?;
                    }
                    write!(self.writer, "{}", evar.name())?;
                }
                self.writer.write_str(" ->\n")?;
                self.indent += 2;
                self.indent()?;
                self.print_expr(expr.handler.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent -= 2;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            Expr::Tuple(t) => {
                self.writer.write_char('{')?;
                self.print_exprs(t.elements.as_slice(), true, false)?;
                self.writer.write_char('}')
            }
            Expr::Values(v) => {
                self.writer.write_char('<')?;
                self.print_exprs(v.values.as_slice(), true, false)?;
                self.writer.write_char('>')
            }
            Expr::Var(var) => write!(self.writer, "{}", var.name()),
        }
    }

    pub fn print_iexpr(&mut self, expr: &IExpr) -> fmt::Result {
        match expr {
            IExpr::Alias(IAlias {
                ref var,
                ref pattern,
                ..
            }) => {
                write!(self.writer, "{} = ", var.name())?;
                self.print_iexpr(pattern)
            }
            IExpr::Apply(apply) => {
                self.writer.write_str("apply ")?;
                self.print_iexprs(apply.callee.as_slice(), true, true)?;
                write!(self.writer, "/{}", apply.args.len())?;
                self.print_iargs(apply.args.as_slice())
            }
            IExpr::Binary(bin) => {
                self.writer.write_str("<<")?;
                for (i, segment) in bin.segments.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    self.print_ibitstring(segment)?;
                }
                self.writer.write_str(">>")
            }
            IExpr::Call(call) => {
                self.writer.write_str("call ")?;
                self.print_iexpr(call.module.as_ref())?;
                self.writer.write_char(':')?;
                self.print_iexpr(call.function.as_ref())?;
                write!(self.writer, "/{}", call.args.len())?;
                self.print_iargs(call.args.as_slice())
            }
            IExpr::Case(expr) => {
                self.writer.write_str("case <")?;
                self.print_iexprs(expr.args.as_slice(), true, false)?;
                self.writer.write_str("> of\n")?;
                self.indent += 2;
                for (i, clause) in expr.clauses.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char('\n')?;
                    }
                    self.print_iclause(clause)?;
                }
                self.writer.write_char('\n')?;
                self.print_iclause(expr.fail.as_ref())?;
                self.indent -= 2;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            IExpr::Catch(expr) => {
                self.writer.write_str("catch ")?;
                self.print_iexprs(expr.body.as_slice(), true, false)?;
                self.writer.write_str(" end")
            }
            IExpr::Cons(cons) => {
                self.writer.write_str("[")?;
                self.print_iexpr(cons.head.as_ref())?;
                self.writer.write_str(" | ")?;
                self.print_iexpr(cons.tail.as_ref())?;
                self.writer.write_str("]")
            }
            IExpr::Exprs(iexprs) => {
                for (i, body) in iexprs.bodies.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(",\n")?;
                        self.indent()?;
                    }
                    self.writer.write_str("do ")?;
                    self.print_iexprs(body.as_slice(), false, false)?;
                    self.indent()?;
                    self.writer.write_str("end\n")?;
                }
                Ok(())
            }
            IExpr::Fun(ref ifun) => {
                self.indent += 2;
                self.print_ifun(ifun)?;
                self.indent -= 2;
                Ok(())
            }
            IExpr::If(expr) => {
                self.writer.write_str("if ")?;
                self.print_iexprs(expr.guards.as_slice(), true, false)?;
                self.writer.write_char('\n')?;
                self.indent += 2;
                self.indent()?;
                self.writer.write_str("then ")?;
                self.print_iexprs(expr.then_body.as_slice(), false, false)?;
                self.indent()?;
                self.writer.write_str("else ")?;
                self.print_iexprs(expr.else_body.as_slice(), false, false)?;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            IExpr::LetRec(expr) => {
                self.writer.write_str("letrec")?;
                self.indent += 2;
                for (i, (binding, expr)) in expr.defs.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(",\n")?;
                    } else {
                        self.writer.write_char('\n')?;
                    }
                    self.indent()?;
                    write!(self.writer, "{} = ", binding.name())?;
                    self.indent += 2;
                    self.print_iexpr(expr)?;
                    self.indent -= 2;
                }
                self.writer.write_char('\n')?;
                self.indent -= 2;
                self.indent()?;
                self.writer.write_str("in ")?;
                self.print_iexprs(expr.body.as_slice(), false, false)?;
                self.indent()?;
                self.writer.write_str("end")
            }
            IExpr::Literal(ref lit) => printing::print_literal(self.writer, lit),
            IExpr::Match(expr) => {
                // match <pattern> = <arg>
                //   when <guard>
                //   else <fail>
                self.writer.write_str("match ")?;
                self.print_iexpr(expr.pattern.as_ref())?;
                self.writer.write_str(" = ")?;
                self.indent += 2;
                self.print_iexpr(expr.arg.as_ref())?;
                self.writer.write_str("\n")?;
                self.indent()?;
                if !expr.guards.is_empty() {
                    self.writer.write_str("when ")?;
                    self.print_iexprs(expr.guards.as_slice(), false, false)?;
                    self.indent()?;
                }
                self.writer.write_str("else\n")?;
                self.print_iclause(expr.fail.as_ref())?;
                self.indent -= 2;
                Ok(())
            }
            IExpr::Map(map) => {
                let print_arg = match map.arg.as_ref() {
                    IExpr::Literal(Literal {
                        value: Lit::Map(m), ..
                    }) if m.is_empty() => false,
                    _ => true,
                };
                if print_arg {
                    self.print_iexpr(map.arg.as_ref())?;
                }
                self.writer.write_str("#{")?;
                for (i, pair) in map.pairs.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    self.indent += 2;
                    self.print_iexprs(pair.key.as_slice(), true, true)?;
                    match pair.op {
                        MapOp::Exact => self.writer.write_str(" := ")?,
                        _ => self.writer.write_str(" => ")?,
                    }
                    self.print_iexpr(pair.value.as_ref())?;
                    self.indent -= 2;
                }
                self.writer.write_char('}')
            }
            IExpr::PrimOp(op) => {
                write!(self.writer, "primop {}", op.name)?;
                self.print_iargs(op.args.as_slice())
            }
            IExpr::Protect(expr) => {
                self.writer.write_str("protect\n")?;
                self.print_iexprs(expr.body.as_slice(), false, false)?;
                self.indent()?;
                self.writer.write_str("end")
            }
            IExpr::Receive1(recv) => {
                self.writer.write_str("receive\n")?;
                self.indent += 2;
                for (i, clause) in recv.clauses.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char('\n')?;
                    }
                    self.print_iclause(clause)?;
                }
                self.indent -= 2;
                self.writer.write_char('\n')?;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&recv.annotations)
            }
            IExpr::Receive2(recv) => {
                self.writer.write_str("receive\n")?;
                self.indent += 2;
                for (i, clause) in recv.clauses.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char('\n')?;
                    }
                    self.print_iclause(clause)?;
                }
                self.indent -= 2;
                if !recv.clauses.is_empty() {
                    self.writer.write_char('\n')?;
                }
                self.indent()?;
                self.writer.write_str("after\n")?;
                self.indent += 2;
                self.indent()?;
                self.print_iexpr(recv.timeout.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent += 2;
                self.indent()?;
                self.print_iexprs(recv.action.as_slice(), false, false)?;
                self.indent -= 4;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&recv.annotations)
            }
            IExpr::Set(set) => {
                write!(self.writer, "set {} = ", set.var.name())?;
                self.print_iexpr(set.arg.as_ref())
            }
            IExpr::Simple(e) => self.print_iexpr(e.expr.as_ref()),
            IExpr::Tuple(t) => {
                self.writer.write_char('{')?;
                self.print_iexprs(t.elements.as_slice(), true, false)?;
                self.writer.write_char('}')
            }
            IExpr::Try(expr) => {
                self.writer.write_str("try <")?;
                self.print_iexprs(expr.args.as_slice(), true, false)?;
                self.writer.write_str("> of\n")?;
                self.indent += 2;
                self.indent()?;
                self.writer.write_char('<')?;
                for (i, v) in expr.vars.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_str(", ")?;
                    }
                    write!(self.writer, "{}", v.name())?;
                }
                self.writer.write_str("> -> ")?;
                self.print_iexprs(expr.body.as_slice(), false, false)?;
                self.indent()?;
                self.writer.write_str("catch ")?;
                for (i, evar) in expr.evars.iter().enumerate() {
                    if i > 0 {
                        self.writer.write_char(':')?;
                    }
                    write!(self.writer, "{}", evar.name())?;
                }
                self.writer.write_str(" ->\n")?;
                self.indent += 2;
                self.print_iexpr(expr.handler.as_ref())?;
                self.writer.write_char('\n')?;
                self.indent -= 2;
                self.indent()?;
                self.writer.write_str("end\n")?;
                self.print_annotations(&expr.annotations)
            }
            IExpr::Var(v) => write!(self.writer, "{}", v.name()),
        }
    }

    fn print_clause(&mut self, clause: &Clause) -> fmt::Result {
        self.indent()?;
        self.writer.write_str("clause ")?;
        self.print_args(clause.patterns.as_slice())?;
        match clause.guard.as_deref() {
            None => self.writer.write_str(" ->\n")?,
            Some(guard) => {
                self.writer.write_str(" when ")?;
                self.print_expr(guard)?;
                self.writer.write_str(" ->\n")?;
            }
        }
        self.indent += 2;
        self.indent()?;
        self.print_expr(clause.body.as_ref())?;
        self.writer.write_char('\n')?;
        self.indent -= 2;
        self.indent()?;
        self.writer.write_str("end\n")?;
        self.print_annotations(&clause.annotations)
    }

    fn print_iclause(&mut self, clause: &IClause) -> fmt::Result {
        self.indent()?;
        self.writer.write_str("clause ")?;
        self.print_iargs(clause.patterns.as_slice())?;
        if clause.guards.is_empty() {
            self.writer.write_str(" -> ")?;
        } else {
            self.writer.write_str(" when ")?;
            self.print_iexprs(clause.guards.as_slice(), true, false)?;
            self.writer.write_str(" -> ")?;
        }
        self.print_iexprs(clause.body.as_slice(), false, false)?;
        self.indent()?;
        self.writer.write_str("end\n")?;
        self.print_annotations(&clause.annotations)
    }

    fn print_bitstring(&mut self, bs: &super::Bitstring) -> fmt::Result {
        self.print_expr(bs.value.as_ref())?;
        if let Some(sz) = bs.size.as_deref() {
            self.writer.write_char(':')?;
            self.print_expr(sz)?;
        }
        self.writer.write_char('/')?;
        self.print_bitspec(bs.spec)
    }

    fn print_ibitstring(&mut self, bs: &IBitstring) -> fmt::Result {
        self.print_iexpr(bs.value.as_ref())?;
        if !bs.size.is_empty() {
            self.writer.write_char(':')?;
            self.print_iexprs(bs.size.as_slice(), true, true)?;
        }
        self.writer.write_char('/')?;
        self.print_bitspec(bs.spec)
    }

    fn print_bitspec(&mut self, spec: BinaryEntrySpecifier) -> fmt::Result {
        match spec {
            BinaryEntrySpecifier::Integer {
                signed,
                endianness,
                unit,
            } => {
                self.writer.write_str("integer")?;
                if signed {
                    self.writer.write_str("-signed")?;
                }
                if endianness != Endianness::Big {
                    write!(self.writer, "-{}", endianness)?;
                }
                if unit != 1 {
                    write!(self.writer, "-unit({})", unit)?;
                }
            }
            BinaryEntrySpecifier::Float { endianness, unit } => {
                self.writer.write_str("float")?;
                if endianness != Endianness::Big {
                    write!(self.writer, "-{}", endianness)?;
                }
                if unit != 1 {
                    write!(self.writer, "-unit({})", unit)?;
                }
            }
            BinaryEntrySpecifier::Binary { unit } if unit == 8 => {
                self.writer.write_str("binary")?;
            }
            BinaryEntrySpecifier::Binary { unit } if unit == 1 => {
                self.writer.write_str("bitstring")?;
            }
            BinaryEntrySpecifier::Binary { unit } => {
                write!(self.writer, "bitstring-unit({})", unit)?;
            }
            BinaryEntrySpecifier::Utf8 => {
                self.writer.write_str("utf8")?;
            }
            BinaryEntrySpecifier::Utf16 { endianness } => {
                self.writer.write_str("utf16")?;
                if endianness != Endianness::Big {
                    write!(self.writer, "-{}", endianness)?;
                }
            }
            BinaryEntrySpecifier::Utf32 { endianness } => {
                self.writer.write_str("utf32")?;
                if endianness != Endianness::Big {
                    write!(self.writer, "-{}", endianness)?;
                }
            }
        }
        Ok(())
    }

    fn print_annotations(&mut self, annos: &Annotations) -> fmt::Result {
        if annos.is_empty() {
            return Ok(());
        }
        self.indent()?;
        self.writer.write_str("-| [")?;
        for (i, (sym, value)) in annos.iter().enumerate() {
            if i > 0 {
                self.writer.write_str(", ")?;
                printing::print_annotation(self.writer, sym, value)?;
            } else {
                printing::print_annotation(self.writer, sym, value)?;
            }
        }
        self.writer.write_str("]\n")?;
        self.indent()
    }

    fn indent(&mut self) -> fmt::Result {
        if self.indent == 0 {
            return Ok(());
        }
        for _ in 0..self.indent {
            self.writer.write_str("  ")?;
        }
        Ok(())
    }
}
