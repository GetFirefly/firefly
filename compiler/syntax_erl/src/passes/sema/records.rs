use std::collections::HashMap;

use firefly_intern::Ident;
use firefly_util::diagnostics::*;

use crate::ast::*;

pub fn analyze_record(diagnostics: &DiagnosticsHandler, module: &mut Module, mut record: Record) {
    let name = record.name.name;

    if let Some(prev) = module.records.get(&name) {
        diagnostics
            .diagnostic(Severity::Error)
            .with_message("record already defined")
            .with_primary_label(record.span, "duplicate definition occurs here")
            .with_secondary_label(prev.span, "previously defined here")
            .emit();
        return;
    }

    let mut field_idx_map = HashMap::<Ident, usize>::new();
    let mut index = 0;
    let mut fields = Vec::<RecordField>::with_capacity(record.fields.len());
    for mut field in record.fields.drain(0..) {
        if field.value.is_none() {
            field.value = Some(atom!(field.name.span, undefined));
        }
        if let Some(prev_idx) = field_idx_map.get(&field.name) {
            let prev = fields.get(*prev_idx).unwrap();
            diagnostics
                .diagnostic(Severity::Error)
                .with_message("duplicate field in record")
                .with_primary_label(field.name.span, "duplicate field occurs here")
                .with_secondary_label(prev.span, "originally defined here")
                .emit();
        } else {
            field_idx_map.insert(field.name, index);
            fields.push(field);
            index += 1;
        }
    }
    record.fields = fields;
    module.records.insert(name, record);
}
