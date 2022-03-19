use std::collections::HashMap;

use liblumen_intern::Ident;

use crate::ast::*;

use super::*;

impl SemanticAnalysis {
    pub(super) fn analyze_record(&mut self, module: &mut Module, mut record: Record) {
        let name = record.name.name;

        if let Some(prev) = module.records.get(&name) {
            self.show_error(
                "record already defined",
                &[
                    (record.span, "duplicate definition occurs here"),
                    (prev.span, "previously defined here"),
                ],
            );
            return;
        }

        let mut field_idx_map = HashMap::<Ident, usize>::new();
        let mut index = 0;
        let mut fields = Vec::<RecordField>::with_capacity(record.fields.len());
        for mut field in record.fields.drain(0..) {
            if field.value.is_none() {
                field.value = Some(atom!(undefined));
            }
            if let Some(prev_idx) = field_idx_map.get(&field.name) {
                let prev = fields.get(*prev_idx).unwrap();
                self.show_error(
                    "duplicate field in record",
                    &[
                        (field.name.span, "duplicate field occurs here"),
                        (prev.span, "originally defined here"),
                    ],
                );
            } else {
                field_idx_map.insert(field.name, index);
                fields.push(field);
                index += 1;
            }
        }
        record.fields = fields;
        module.records.insert(name, record);
    }
}
