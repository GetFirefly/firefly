mod attributes;
mod functions;
mod inject;
mod records;
mod verify;

use std::collections::{BTreeMap, HashMap, HashSet};

use liblumen_diagnostics::*;
use liblumen_pass::Pass;

use crate::ast::{self, *};

use self::inject::{AddAutoImports, DefinePseudoLocals};
use self::verify::{VerifyOnLoadFunctions, VerifyTypeSpecs};

/// This pass is responsible for taking a set of top-level forms and
/// analyzing them in the context of a new module to produce a fully
/// constructed and initially validated module.
///
/// Some of the analyses run include:
///
/// * If configured to do so, warns if functions are missing type specs
/// * Warns about type specs for undefined functions
/// * Warns about redefined attributes
/// * Errors on invalid syntax in built-in attributes (e.g. -import(..))
/// * Errors on mismatched function clauses (name/arity)
/// * Errors on unterminated function clauses
/// * Errors on redefined functions
///
/// And a few other similar lints
pub struct SemanticAnalysis {
    reporter: Reporter,
    span: SourceSpan,
    name: Ident,
}
impl SemanticAnalysis {
    pub fn new(reporter: Reporter, span: SourceSpan, name: Ident) -> Self {
        Self {
            reporter,
            span,
            name,
        }
    }
}
impl Pass for SemanticAnalysis {
    type Input<'a> = Vec<TopLevel>;
    type Output<'a> = ast::Module;

    fn run<'a>(&mut self, mut forms: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut module = Module {
            span: self.span,
            name: self.name,
            vsn: None,
            author: None,
            on_load: None,
            compile: None,
            imports: HashMap::new(),
            exports: HashSet::new(),
            removed: HashMap::new(),
            types: HashMap::new(),
            exported_types: HashSet::new(),
            specs: HashMap::new(),
            behaviours: HashSet::new(),
            callbacks: HashMap::new(),
            records: HashMap::new(),
            attributes: HashMap::new(),
            functions: BTreeMap::new(),
            deprecation: None,
            deprecations: HashSet::new(),
        };

        for form in forms.drain(0..) {
            match form {
                TopLevel::Attribute(attr) => self.analyze_attribute(&mut module, attr),
                TopLevel::Record(record) => self.analyze_record(&mut module, record),
                TopLevel::Function(function) => self.analyze_function(&mut module, function),
                _ => panic!("unexpected top-level form: {:?}", &form),
            }
        }

        let mut passes = AddAutoImports
            //.chain(DefinePseudoLocals)
            .chain(VerifyOnLoadFunctions::new(self.reporter.clone()))
            .chain(VerifyTypeSpecs::new(self.reporter.clone()));

        passes.run(&mut module)?;

        Ok(module)
    }
}
impl SemanticAnalysis {
    fn show_error(&mut self, message: &str, labels: &[(SourceSpan, &str)]) {
        if labels.is_empty() {
            self.reporter
                .diagnostic(Diagnostic::error().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.reporter.diagnostic(
                Diagnostic::error()
                    .with_message(message)
                    .with_labels(labels),
            );
        }
    }

    fn show_warning(&mut self, message: &str, labels: &[(SourceSpan, &str)]) {
        if labels.is_empty() {
            self.reporter
                .diagnostic(Diagnostic::warning().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.reporter.diagnostic(
                Diagnostic::error()
                    .with_message(message)
                    .with_labels(labels),
            );
        }
    }
}
