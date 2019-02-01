use crate::parser::ast::*;

use super::ImmutableVisitor;

/// A visitor that can be used to convert an AST back into source code.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PrettyPrintVisitor {
    output: String,
}

impl PrettyPrintVisitor {
    pub fn new() -> Self {
        PrettyPrintVisitor {
            output: String::new(),
        }
    }

    pub fn get_output(&self) -> &String {
        &self.output
    }

    fn stringify_atom(&self, atom: &Ident) -> String {
        atom.name.as_str().get().to_owned()
    }
}

impl<'ast> ImmutableVisitor<'ast> for PrettyPrintVisitor {
    fn visit(&mut self, module: &'ast Module) {
        self.output.push_str(&format!(
            "-module({}).\n\n",
            self.stringify_atom(&module.name)
        ));

        // TODO
    }
}
