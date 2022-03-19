use std::io::Write;

use super::types::*;

const DIALECT_TPL: &'static str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tablegen/templates/dialect.rs.hbs"
);
const OPERATION_TPL: &'static str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tablegen/templates/operation.rs.hbs"
);
const ATTRIBUTE_TPL: &'static str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tablegen/templates/attribute.rs.hbs"
);
const TYPE_TPL: &'static str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tablegen/templates/type.rs.hbs"
);
const ENUM_TPL: &'static str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tablegen/templates/enum.rs.hbs"
);
const BITFLAG_TPL: &'static str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tablegen/templates/bitflag.rs.hbs"
);

use handlebars::handlebars_helper;
use heck::{ToPascalCase, ToSnakeCase};
use serde_json::{map::Map, Value};

handlebars_helper!(to_pascal_case: |s: str| s.to_pascal_case());
handlebars_helper!(to_snake_case: |s: str| s.to_snake_case());
handlebars_helper!(to_rust_type: |s: str| to_rust_type_impl(s));
handlebars_helper!(to_enum_value_type: |s: object| to_enum_value_type_impl(s));
handlebars_helper!(to_enum_return_type: |s: object| to_enum_return_type_impl(s));
handlebars_helper!(to_rust_identifier: |s: str| to_rust_identifier_impl(s));
handlebars_helper!(is_variadic_type: |s: str| is_variadic_type_impl(s));
handlebars_helper!(strip_namespace: |s: str| strip_cpp_namespace(s).to_owned());

/// Generates Rust code from the given Dialect, writing output to `writer`
pub fn to_rust<W>(dialect: Dialect, writer: W) -> anyhow::Result<()>
where
    W: Write,
{
    use handlebars::Handlebars;
    use serde_json::*;

    let mut handlebars = Handlebars::new();
    handlebars.source_map_enabled(true);
    handlebars.set_strict_mode(true);
    handlebars.register_template_file("dialect", DIALECT_TPL)?;
    handlebars.register_template_file("operation", OPERATION_TPL)?;
    handlebars.register_template_file("attribute", ATTRIBUTE_TPL)?;
    handlebars.register_template_file("type", TYPE_TPL)?;
    handlebars.register_template_file("enum", ENUM_TPL)?;
    handlebars.register_template_file("enum", BITFLAG_TPL)?;
    // Do not escape strings
    handlebars.register_escape_fn(|s: &str| s.to_owned());
    handlebars.register_helper("to_pascal_case", Box::new(to_pascal_case));
    handlebars.register_helper("to_snake_case", Box::new(to_snake_case));
    handlebars.register_helper("to_rust_type", Box::new(to_rust_type));
    handlebars.register_helper("to_enum_value_type", Box::new(to_enum_value_type));
    handlebars.register_helper("to_enum_return_type", Box::new(to_enum_return_type));
    handlebars.register_helper("to_rust_identifier", Box::new(to_rust_identifier));
    handlebars.register_helper("is_variadic_type", Box::new(is_variadic_type));
    handlebars.register_helper("strip_namespace", Box::new(strip_namespace));

    // Modify the context a bit to simplify the templates
    let builder_name = format!("{}Builder", dialect.name.to_pascal_case());
    let mut context = serde_json::to_value(dialect)?;
    {
        let dialect_obj = context.as_object_mut().unwrap();
        dialect_obj.insert("builder_name".to_owned(), Value::String(builder_name));
    }

    handlebars.render_to_write("dialect", &context, writer)?;
    Ok(())
}

#[inline]
fn strip_cpp_namespace(s: &str) -> &str {
    s.trim_start_matches("::").split("::").last().unwrap()
}

// Converts an FFI type name to a Rust type name
fn to_rust_type_impl(ty: &str) -> String {
    if ty.starts_with("::llvm") {
        return llvm_type_to_rust_type(ty);
    }
    if ty.starts_with("::mlir") {
        return mlir_type_to_rust_type(ty);
    }
    match ty {
        "bool" => "bool".to_owned(),
        "uint8_t" => "u8".to_owned(),
        "uint16_t" => "u16".to_owned(),
        "uint32_t" => "u32".to_owned(),
        "uint64_t" => "u64".to_owned(),
        unknown => panic!("unrecognized type: {}", unknown),
    }
}

fn to_enum_value_type_impl(ty: &Map<String, Value>) -> String {
    let underlying_type = ty.get("underlyingType").and_then(|v| v.as_str()).unwrap();

    match underlying_type {
        "uint8_t" => "u8".to_owned(),
        "uint16_t" => "u16".to_owned(),
        "uint32_t" => "u32".to_owned(),
        "uint64_t" => "u64".to_owned(),
        "" => {
            let storage_type = ty.get("storage_type").and_then(|v| v.as_str()).unwrap();
            match storage_type {
            "::mlir::StringAttr" => "u32".to_owned(),
                other => panic!("invalid enum value type, unspecified underlying type and non-string storage type ({})", other),
            }
        }
        other => panic!(
            "invalid enum value type, unknown underlying type: {}",
            other
        ),
    }
}

fn to_enum_return_type_impl(ty: &Map<String, Value>) -> String {
    let value_ty = to_enum_value_type_impl(ty);
    let return_type = ty.get("returnType").and_then(|v| v.as_str()).unwrap();
    let class_name = ty.get("className").and_then(|v| v.as_str()).unwrap();

    if return_type.starts_with("::lumen::eir") {
        // This is the enumeration type itself, so we can use the class name of this enum
        // I'm unaware of any situations where this isn't the case, so assert here if one ever comes
        // up so we can handle it properly
        let stripped = return_type.strip_prefix("::lumen::eir::").unwrap();
        assert_eq!(stripped, class_name);
        class_name.to_owned()
    } else {
        value_ty
    }
}

fn is_variadic_type_impl(ty: &str) -> bool {
    return ty.starts_with("::llvm::ArrayRef");
}

fn llvm_type_to_rust_type(ty: &str) -> String {
    match parse_generic_ty(ty) {
        Ok(("::llvm::Optional", inner)) => format!("Option<{}>", &to_rust_type_impl(inner)),
        Ok(("::llvm::ArrayRef", inner)) => format!("&[{}]", &to_rust_type_impl(inner)),
        Err("::llvm::StringRef") => "StringRef".to_owned(),
        Err("::llvm::APInt") => "::liblumen_llvm::APInt".to_owned(),
        Err("::llvm::APFloat") => "::liblumen_llvm::APFloat".to_owned(),
        Ok((_, _)) => panic!("unsupported use of generic llvm type: {}", ty),
        Err(unknown) => panic!("unrecognized llvm type: {}", unknown),
    }
}

fn mlir_type_to_rust_type(ty: &str) -> String {
    match ty {
        "::mlir::Type" => "::liblumen_mlir::ir::Type".to_owned(),
        "::mlir::Block" => "::liblumen_mlir::ir::Block".to_owned(),
        "::mlir::Region" => "::liblumen_mlir::ir::Region".to_owned(),
        "::mlir::Value" => "::liblumen_mlir::ir::Value".to_owned(),
        unknown => panic!("unrecognized mlir type: {}", unknown),
    }
}

fn parse_generic_ty(ty: &str) -> Result<(&str, &str), &str> {
    if let Some(lhs) = ty.find('<') {
        if let Some(rhs) = ty.rfind('>') {
            return Ok((&ty[0..lhs], &ty[(lhs + 1)..rhs]));
        }
    }
    Err(ty)
}

// Converts the given string to a valid Rust identifier
#[inline]
fn to_rust_identifier_impl(ident: &str) -> String {
    ident.to_snake_case().replace('.', "_")
}
