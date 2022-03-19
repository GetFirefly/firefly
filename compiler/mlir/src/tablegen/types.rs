use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

// Used during deserialization for references to a definition
#[derive(Serialize, Deserialize)]
pub struct EntityRef {
    #[serde(rename(deserialize = "def"))]
    name: String,
    kind: String,
}

#[derive(Serialize, Deserialize)]
pub struct Dialect {
    #[serde(rename(deserialize = "name"))]
    pub mnemonic: String,
    #[serde(rename(deserialize = "!name"))]
    pub name: String,
    pub summary: String,
    pub cpp_namespace: String,
    #[serde(rename(deserialize = "dependentDialects"))]
    pub dependent_dialects: Vec<String>,
    pub attributes: Vec<AttrDef>,
    pub types: Vec<TypeDef>,
    pub enums: Vec<Enum>,
    pub bitflags: Vec<Enum>,
    pub operations: Vec<Op>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all(deserialize = "camelCase"))]
pub struct AttrDef {
    #[serde(rename(deserialize = "!name"))]
    pub name: String,
    pub dialect: EntityRef,
    pub mnemonic: String,
    pub cpp_class_name: String,
    pub cpp_type: String,
    pub default_value: Option<String>,
    pub parameters: Parameters,
    pub storage_type: String,
    pub is_optional: bool,
    pub traits: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all(deserialize = "camelCase"))]
pub struct TypeDef {
    #[serde(rename(deserialize = "!name"))]
    pub name: String,
    pub mnemonic: String,
    pub builder_call: String,
    pub cpp_class_name: String,
    pub cpp_type: String,
    pub dialect: EntityRef,
    pub parameters: Parameters,
    pub summary: String,
    pub traits: Vec<EntityRef>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all(deserialize = "camelCase"))]
pub struct Enum {
    #[serde(rename(deserialize = "!name"))]
    pub name: String,
    pub summary: String,
    pub class_name: String,
    pub const_builder_call: Option<String>,
    pub convert_from_storage: Option<String>,
    pub cpp_namespace: String,
    pub default_value: Option<String>,
    #[serde(rename(deserialize = "enumerants"))]
    pub variants: EnumVariant,
    pub llvm_class_name: Option<String>,
    pub return_type: String,
    pub storage_type: String,
    pub string_to_symbol_fn_name: String,
    pub symbol_to_string_fn_name: String,
    pub symbol_to_string_fn_ret_type: String,
    pub underlying_to_symbol_fn_name: String,
    pub underlying_type: String,
    pub value_type: EntityRef,
    pub valid_bits: u8,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all(deserialize = "camelCase"))]
pub struct EnumVariant {
    #[serde(rename(deserialize = "!name"))]
    pub name: String,
    pub const_builder_call: String,
    pub convert_from_storage: String,
    pub cpp_namespace: String,
    pub default_value: Option<String>,
    pub llvm_enumerant: Option<String>,
    pub return_type: String,
    pub storage_type: String,
    pub str: Option<String>,
    pub symbol: String,
    pub value: usize,
    pub value_type: EntityRef,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Op {
    #[serde(rename(deserialize = "!name"))]
    pub name: String,
    #[serde(rename(deserialize = "opName"))]
    pub mnemonic: String,
    #[serde(rename(deserialize = "!superclasses"))]
    pub superclasses: Vec<String>,
    #[serde(rename(deserialize = "opDialect"))]
    pub dialect: EntityRef,
    pub arguments: Parameters,
    pub regions: Parameters,
    pub results: Parameters,
    pub successors: Parameters,
    pub summary: String,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Parameters {
    Dag(DagParameters),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "def", rename_all = "lowercase")]
pub enum DagOperator {
    Ins,
    Outs,
    Region,
    Successor,
}

#[derive(Serialize, Deserialize)]
pub struct DagParameters {
    operator: DagOperator,
    args: Vec<DagParametersArg>,
}

/*
 [
   {
     "def": "AtomRefParameter",
     "kind": "def",
     "printable": "AtomRefParameter"
   },
   "value"
 ]
*/
#[derive(Serialize)]
pub struct DagParametersArg {
    name: String,
    ty: EntityRef,
}
impl<'de> Deserialize<'de> for DagParametersArg {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(DagParametersArgVisitor::new())
    }
}

struct DagParametersArgVisitor {
    marker: core::marker::PhantomData<fn() -> DagParametersArg>,
}
impl DagParametersArgVisitor {
    fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}
impl<'de> serde::de::Visitor<'de> for DagParametersArgVisitor {
    type Value = DagParametersArg;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "a dag parameter arg")
    }

    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: serde::de::SeqAccess<'de>,
    {
        use serde::de;
        let ty: EntityRef = access
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let name: String = access
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;

        Ok(DagParametersArg { name, ty })
    }
}
