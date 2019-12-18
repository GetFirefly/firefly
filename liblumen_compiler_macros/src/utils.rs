use std::string::ToString;

use syn::{Attribute, Lit, Meta, MetaList, MetaNameValue, NestedMeta};

/// Extracts all of the documentation comments from the set of attributes on an item
///
/// This combines all docs into a single `String`, joined by newlines.
///
/// If no docs are present, `None` is returned.
pub fn extract_docs(attributes: &[Attribute]) -> Option<String> {
    let doc_items = find_attrs(attributes, "doc")
        .filter_map(get_name_value_meta_lit)
        .map(|l| {
            let s = lit_to_string(l).unwrap();
            let trimmed = s.as_str().trim_end();
            trimmed.to_string()
        })
        .collect::<Vec<_>>();
    Some(doc_items.as_slice().join("\n"))
}

/// Returns an iterator of attribute metadata for attributes on the current item with the given name
pub fn find_attrs<'a>(
    attributes: &'a [Attribute],
    attr_name: &'static str,
) -> impl Iterator<Item = Meta> + 'a {
    attributes
        .iter()
        .filter_map(with_name(attr_name, try_get_named_attribute))
}

/// Returns the first matching attribute by name, removing it from the source attribute vector
pub fn pop_attr<'a>(attributes: &mut Vec<Attribute>, attr_name: &'static str) -> Option<Attribute> {
    let mut found = None;
    for (i, attr) in attributes.iter().enumerate() {
        match attr.parse_meta() {
            Ok(ref meta) if is_named_meta(meta, attr_name) => {
                found = Some(i);
                break;
            }
            _ => continue,
        }
    }
    if let Some(index) = found {
        Some(attributes.remove(index))
    } else {
        None
    }
}

// Convenience function for wrapping a callback that requires an attribute name
#[inline]
fn with_name<'a, T, U, F>(name: &'static str, fun: F) -> impl Fn(T) -> U + 'a
where
    F: 'a + Fn(T, &'static str) -> U,
{
    move |t| fun(t, name)
}

// Extracts the attribute metadata from the given attribute, if it matches the given name
fn try_get_named_attribute(attr: &Attribute, attr_name: &'static str) -> Option<Meta> {
    match attr.parse_meta() {
        Err(_) => None,
        Ok(meta) => try_get_named_meta(meta, attr_name),
    }
}

// Extracts the inner `Meta` value from a `NestedMeta` value, if it is of meta type
pub(crate) fn try_get_nested_meta(nested: &NestedMeta) -> Option<Meta> {
    match nested {
        NestedMeta::Meta(meta) => Some(meta.clone()),
        _ => None,
    }
}

// Returns `Some(meta)` if `meta` has the given name, otherwise `None`
pub(crate) fn try_get_named_meta(meta: Meta, attr_name: &'static str) -> Option<Meta> {
    if is_named_meta(&meta, attr_name) {
        return Some(meta);
    }
    None
}

pub(crate) fn named_meta_to_map_entry(meta: Meta) -> Option<(String, Meta)> {
    match meta {
        Meta::List(MetaList { ref path, .. }) => {
            let name = path.get_ident()?.to_string();
            Some((name, meta))
        }
        Meta::NameValue(MetaNameValue { ref path, .. }) => {
            let name = path.get_ident()?.to_string();
            Some((name, meta))
        }
        _ => None,
    }
}

// Returns `true` if the given `Meta` value has the given name
fn is_named_meta(meta: &Meta, attr_name: &'static str) -> bool {
    match meta {
        &Meta::Path(ref path) if path.is_ident(attr_name) => true,
        &Meta::List(MetaList { ref path, .. }) if path.is_ident(attr_name) => true,
        &Meta::NameValue(MetaNameValue { ref path, .. }) if path.is_ident(attr_name) => true,
        _ => false,
    }
}

// Returns the inner `Lit` value, if the given `Meta` value is of name/value type
fn get_name_value_meta_lit(meta: Meta) -> Option<Lit> {
    match meta {
        Meta::NameValue(MetaNameValue { lit, .. }) => Some(lit),
        _ => None,
    }
}

// Maps a `Lit` value to `String`, if it represents a literal string
pub fn lit_to_string(lit: Lit) -> Option<String> {
    match lit {
        Lit::Str(s) => Some(s.value()),
        _ => None,
    }
}
