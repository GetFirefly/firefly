use std::collections::HashMap;

use proc_macro2::{Ident, Span};

use quote::quote_spanned;

use syn::parse::{Parse, ParseStream};
use syn::parse_quote;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::token;
use syn::visit_mut::VisitMut;
use syn::{Attribute, AttributeArgs};
use syn::{Error, Result};
use syn::{Expr, ExprLit};
use syn::{Lit, LitBool, LitInt, LitStr};
use syn::{Meta, MetaList, MetaNameValue, NestedMeta};

use crate::utils;

struct DisableDocTests;
impl VisitMut for DisableDocTests {
    fn visit_attribute_mut(&mut self, attr: &mut Attribute) {
        if attr.path.is_ident("doc") {
            *attr = parse_quote! { #[doc = ""] };
        }
    }
}

// Parses an options struct with attributes.
//
//     #[option_group(name = "codegen", short = "C", help = "Set a codegen option")]
//     struct CodeGenOptions {
//         #[option]
//         linker: Option<PathBuf>,
//         #[option(allow_multiple(true))]
//         linker_arg: Option<Vec<String>>,
//     }
//
// An option group is converted to `clap::App` which expects to receive arguments like so:
//
//     -<short> help
//     --<name> help
//
//     -<short> OPT[=VAL]...
//     --<name> OPT[=VAL]...
//
// If `help` is given, all of the options will be printed with their documentation.
//
// An option allows controlling the parsing of individual options, however there are a few
// caveats to be aware of:
//
// - All options which are `takes_value(true)`, are forced to `require_equals(true)`
//   - This is done for you, not needed to manually specify
//   - This forces all opt/val pairs to be given in the form `OPT=VAL`, i.e. `OPT VAL` is not
//     permitted
// - For options which can be specified multiple times, but only one value per time:
//   - Must specify `number_of_values(1)`
// - For options which can have multiple values, but only occur once:
//   - Do _not_ specify `multiple(true)`
//   - Do specify `require_delimiter(true)`, comma by default, use `value_delimiter` to change
// - For options which accept flag-like values:
//   - For options which accept multiple flags at once, `multiple(true)` cannot be used, arguments
//     must be quoted
//   - For options which accept only a single flag at once, set `multiple(true)` _and_
//     `number_of_values(1)` and require the flag to be quoted
pub struct OptionGroupStruct {
    pub def: syn::ItemStruct,
    pub options: Vec<OptionInfo>,
}
impl quote::ToTokens for OptionGroupStruct {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let mut def = self.def.clone();
        DisableDocTests.visit_item_struct_mut(&mut def);
        def.to_tokens(tokens)
    }
}
impl Parse for OptionGroupStruct {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut parsed = Self {
            def: input.parse()?,
            options: Vec::new(),
        };
        parsed.options.reserve(parsed.def.fields.len());
        for field in parsed.def.fields.iter_mut() {
            if let Some(attr) = utils::pop_attr(&mut field.attrs, "option") {
                let ident = field.ident.clone().unwrap();
                let docs = utils::extract_docs(field.attrs.as_slice());
                let info = OptionInfo::from_meta(ident, docs, attr.parse_meta()?)?;
                parsed.options.push(info);
            } else {
                return Err(Error::new(
                    field.span(),
                    "expected 'option' attribute on this field, but one was not found",
                ));
            }
        }
        Ok(parsed)
    }
}

/// This struct contains the metadata given to the `#[option_group(..)]` attribute
pub struct OptionGroupConfig {
    pub name: String,
    pub short: String,
    pub help: String,
}
impl OptionGroupConfig {
    pub fn from_args(args: AttributeArgs) -> Result<Self> {
        let mut name = None;
        let mut short = None;
        let mut help = None;
        if args.len() == 0 {
            return Err(Error::new(
                Span::call_site(),
                "expected at least one argument",
            ));
        }
        for nested in args.iter() {
            match nested {
                NestedMeta::Meta(Meta::NameValue(MetaNameValue {
                    ref path, ref lit, ..
                })) => {
                    let value = utils::lit_to_string(lit.clone())
                        .ok_or_else(|| Self::invalid_string(lit.span()))?;
                    if path.is_ident("name") {
                        if name.is_some() {
                            return Err(Error::new(path.span(), "tried to set 'help' twice"));
                        }
                        name = Some(value);
                    } else if path.is_ident("short") {
                        if short.is_some() {
                            return Err(Error::new(path.span(), "tried to set 'help' twice"));
                        }
                        short = Some(value);
                    } else if path.is_ident("help") {
                        if help.is_some() {
                            return Err(Error::new(path.span(), "tried to set 'help' twice"));
                        }
                        help = Some(value);
                    } else {
                        return Err(Error::new(
                            path.span(),
                            "unrecognized argument, expected name|short|help",
                        ));
                    }
                }
                NestedMeta::Meta(item @ Meta::List(_)) => {
                    return Err(Error::new(
                        item.span(),
                        "unsupported argument type, use `name = \"value\"` syntax",
                    ));
                }
                item => {
                    return Err(Error::new(
                        item.span(),
                        "unexpected argument for 'option_group' attribute",
                    ))
                }
            }
        }
        let name = name.ok_or_else(|| Self::missing(args[0].span(), "name"))?;
        let short = short.ok_or_else(|| Self::missing(args[0].span(), "short"))?;
        let help = help.ok_or_else(|| Self::missing(args[0].span(), "help"))?;
        Ok(Self { name, short, help })
    }
}
impl OptionGroupConfig {
    fn missing(span: Span, item_name: &'static str) -> Error {
        Error::new(
            span,
            &format!(
                "required option group configuration is missing or invalid: '{}'",
                item_name
            ),
        )
    }

    fn invalid_string(span: Span) -> Error {
        Error::new(span, "expected string value")
    }
}

/// This trait is used to convert values to an expression for use
/// in argument lists and other places where conversion to a
/// `syn::Expr` may be valuable
pub trait ToArg {
    fn to_arg(&self, span: Span) -> Expr;
}
impl ToArg for String {
    fn to_arg(&self, span: Span) -> Expr {
        make_lit_expr(Lit::Str(LitStr::new(self.as_str(), span)))
    }
}
impl ToArg for bool {
    fn to_arg(&self, span: Span) -> Expr {
        make_lit_expr(Lit::Bool(LitBool { value: *self, span }))
    }
}
impl ToArg for u64 {
    fn to_arg(&self, span: Span) -> Expr {
        make_lit_expr(Lit::Int(LitInt::new(&format!("{}", *self), span)))
    }
}

/// This trait is used to convert values to a puncutated expression
/// for use in argument lists.
pub trait ToArgs {
    fn to_args(&self, span: Span) -> Punctuated<Expr, token::Comma>;
}
impl<T> ToArgs for T
where
    T: ToArg,
{
    fn to_args(&self, span: Span) -> Punctuated<Expr, token::Comma> {
        let mut args = Punctuated::new();
        args.push(self.to_arg(span));
        args
    }
}
impl<T> ToArgs for Vec<T>
where
    T: ToArg,
{
    fn to_args(&self, span: Span) -> Punctuated<Expr, token::Comma> {
        let mut args = Punctuated::new();
        for item in self.iter() {
            args.push(item.to_arg(span.clone()));
        }
        args
    }
}
impl<T> ToArgs for &[T]
where
    T: ToArg,
{
    fn to_args(&self, span: Span) -> Punctuated<Expr, token::Comma> {
        let mut args = Punctuated::new();
        let elems = self.iter().map(|item| item.to_arg(span.clone()));
        let quoted = quote_spanned! { span =>
            &[#(#elems),*]
        };
        args.push(parse_quote!(#quoted));
        args
    }
}

#[inline]
fn make_lit_expr(lit: Lit) -> Expr {
    Expr::Lit(ExprLit {
        attrs: Vec::new(),
        lit,
    })
}

/// This struct defines compile-time metadata about a specific option in an option group
#[derive(Debug)]
pub struct OptionInfo {
    name: Ident,
    docs: Option<String>,
    help: Option<String>,
    default_value: Option<String>,
    conflicts: Option<Vec<String>>,
    takes_value: bool,
    multiple: bool,
    allow_hyphen_values: bool,
    number_of_values: Option<u64>,
    max_values: Option<u64>,
    min_values: Option<u64>,
    possible_values: Option<Vec<String>>,
    require_delimiter: Option<bool>,
    value_delimiter: Option<String>,
    value_name: Option<String>,
    value_names: Option<Vec<String>>,
    next_line_help: Option<bool>,
    hidden: bool,
}
macro_rules! extend_expr {
    ($e:expr, $info:ident . $field:ident) => {
        extend_expr_override_default_name!($e, $info.$field, stringify!($field), &$info.$field)
    };
    ($e:expr, $info:ident . $field:ident, $val:expr) => {
        extend_expr_override_default_name!($e, $info.$field, stringify!($field), $val)
    };
    ($e:expr, $field:ident, $val:expr) => {
        extend_expr_override_default_name!($e, $info.$field, stringify!($field), $val)
    };
}
macro_rules! extend_expr_opt {
    ($e:expr, $info:ident . $field:ident) => {
        extend_expr_opt_override_default_name!($e, $info.$field, stringify!($field), &$info.$field)
    };
    ($e:expr, $info:ident . $field:ident, $val:expr) => {
        extend_expr_opt_override_default_name!($e, $info.$field, stringify!($field), $val)
    };
    ($e:expr, $field:ident, $val:expr) => {
        extend_expr_opt_override_default_name!($e, $info.$field, stringify!($field), $val)
    };
}
macro_rules! extend_expr_opt_override_default_name {
    ($e:expr, $info:ident . $field:ident, $name:expr) => {
        Self::extend_build_arg_opt($e, $name, &$info.$field)
    };
    ($e:expr, $info:ident . $field:ident, $name:expr, $val:expr) => {
        Self::extend_build_arg_opt($e, $name, $val)
    };
}
macro_rules! extend_expr_override_default_name {
    ($e:expr, $info:ident . $field:ident, $name:expr) => {
        Self::extend_build_arg($e, $name, &$info.$field)
    };
    ($e:expr, $info:ident . $field:ident, $name:expr, $val:expr) => {
        Self::extend_build_arg($e, $name, $val)
    };
}
impl OptionInfo {
    pub fn docs(&self) -> Option<&str> {
        if self.help.is_some() {
            self.help.as_ref().map(|h| h.as_str())
        } else {
            self.docs.as_ref().map(|d| d.as_str())
        }
    }

    /// Constructs the AST for the series of calls to construct a `clap::Arg` at
    /// runtime, i.e. the following:
    ///
    /// ```rust,ignore
    /// clap::Arg::with_name("foo")
    ///      .help("Some help documentation")
    ///      .takes_value(true)
    ///      .possible_values(&["bar", "baz"])
    /// ```
    pub fn to_arg(&self) -> Expr {
        let name = self.name.clone();
        let build_arg_expr: Expr = parse_quote! {
            clap::Arg::with_name(stringify!(#name)).long(stringify!(#name))
        };
        // Use explicit 'help' if given, otherwise use whatever doc comments we have
        let build_arg_expr = if self.help.is_some() {
            extend_expr_opt!(build_arg_expr, self.help)
        } else {
            extend_expr_opt_override_default_name!(build_arg_expr, self.docs, "help")
        };
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.default_value);
        let conflicts = self.conflicts.as_ref().map(|dvs| dvs.as_slice());
        let build_arg_expr = extend_expr_opt_override_default_name!(
            build_arg_expr,
            self.conflicts,
            "conflicts_with_all",
            &conflicts
        );
        let build_arg_expr = extend_expr!(build_arg_expr, self.takes_value);
        let build_arg_expr = extend_expr!(build_arg_expr, self.multiple);
        let build_arg_expr = extend_expr!(build_arg_expr, self.allow_hyphen_values);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.number_of_values);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.max_values);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.min_values);
        let possible_values = self.possible_values.as_ref().map(|pvs| pvs.as_slice());
        let build_arg_expr =
            extend_expr_opt!(build_arg_expr, self.possible_values, &possible_values);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.require_delimiter);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.value_delimiter);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.value_name);
        let value_names = self.value_names.as_ref().map(|vns| vns.as_slice());
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.value_names, &value_names);
        let build_arg_expr = extend_expr_opt!(build_arg_expr, self.next_line_help);
        let build_arg_expr = extend_expr!(build_arg_expr, self.hidden);

        build_arg_expr
    }

    pub fn to_option_info_args(&self) -> Punctuated<Expr, token::Comma> {
        let desc_opt = self.docs();
        let mut args = Punctuated::new();
        let span = self.name.span();
        let name = self.name.to_string();
        args.push(Expr::Lit(ExprLit {
            attrs: Vec::new(),
            lit: Lit::Str(LitStr::new(name.as_ref(), span.clone())),
        }));
        let desc_quoted = if desc_opt.is_none() {
            parse_quote!(None)
        } else {
            let desc_lit = Expr::Lit(ExprLit {
                attrs: Vec::new(),
                lit: Lit::Str(LitStr::new(desc_opt.unwrap(), span.clone())),
            });
            parse_quote!(Some(#desc_lit))
        };
        args.push(desc_quoted);
        args
    }

    fn extend_build_arg<T>(build_arg_expr: Expr, method_name: &'static str, value: &T) -> Expr
    where
        T: ToArgs,
    {
        let span = build_arg_expr.span();
        let method = Ident::new(method_name, span.clone());
        let args = value.to_args(span.clone());
        parse_quote!(
            #build_arg_expr
              .#method(#args)
        )
    }

    fn extend_build_arg_opt<T>(
        build_arg_expr: Expr,
        method_name: &'static str,
        value_ref: &Option<T>,
    ) -> Expr
    where
        T: ToArgs,
    {
        match value_ref {
            &None => build_arg_expr,
            &Some(ref value) => Self::extend_build_arg(build_arg_expr, method_name, value),
        }
    }

    /// Constructs an `OptionInfo` from the name, documentation, and attribute meta list
    /// given to the `#[option(..)]` macro.
    pub fn from_meta(name: Ident, docs: Option<String>, meta: Meta) -> Result<Self> {
        if let Meta::List(MetaList { nested, .. }) = meta {
            let opt_metas = nested
                .iter()
                .filter_map(utils::try_get_nested_meta)
                .filter_map(utils::named_meta_to_map_entry)
                .collect::<HashMap<String, Meta>>();

            let default_value = Self::opt_string(&opt_metas, "default_value")?;
            let takes_value = Self::boolean(&opt_metas, "takes_value", default_value.is_some())?;
            Ok(Self {
                name,
                docs,
                help: Self::opt_string(&opt_metas, "help")?,
                default_value,
                conflicts: Self::opt_string_list(&opt_metas, "conflicts")?,
                takes_value,
                multiple: Self::boolean(&opt_metas, "multiple", false)?,
                allow_hyphen_values: Self::boolean(&opt_metas, "allow_hyphen_values", false)?,
                number_of_values: Self::opt_uint(&opt_metas, "number_of_values")?,
                max_values: Self::opt_uint(&opt_metas, "max_values")?,
                min_values: Self::opt_uint(&opt_metas, "min_values")?,
                possible_values: Self::opt_string_list(&opt_metas, "possible_values")?,
                require_delimiter: Self::opt_boolean(&opt_metas, "require_delimiter")?,
                value_delimiter: Self::opt_string(&opt_metas, "value_delimiter")?,
                value_name: Self::opt_string(&opt_metas, "value_name")?,
                value_names: Self::opt_string_list(&opt_metas, "value_names")?,
                next_line_help: Self::opt_boolean(&opt_metas, "next_line_help")?,
                hidden: Self::boolean(&opt_metas, "hidden", false)?,
            })
        } else if let Meta::Path(_) = meta {
            Ok(Self {
                name,
                docs,
                help: None,
                default_value: None,
                conflicts: None,
                takes_value: false,
                multiple: false,
                allow_hyphen_values: false,
                number_of_values: None,
                max_values: None,
                min_values: None,
                possible_values: None,
                require_delimiter: None,
                value_delimiter: None,
                value_name: None,
                value_names: None,
                next_line_help: None,
                hidden: false,
            })
        } else {
            Err(Error::new(
                meta.span(),
                "the 'option' attribute must be of the form `#[option(..)]`",
            ))
        }
    }

    fn opt_string(metas: &HashMap<String, Meta>, key: &'static str) -> Result<Option<String>> {
        with_optional_meta_value(metas, key, |mlist| {
            if mlist.nested.len() != 1 {
                return Err(Error::new(
                    mlist.span(),
                    "expected a single string value here",
                ));
            }
            expect_literal_string(mlist.nested.first().unwrap()).map(|s| Some(s))
        })
    }

    fn opt_string_list(
        metas: &HashMap<String, Meta>,
        key: &'static str,
    ) -> Result<Option<Vec<String>>> {
        with_optional_meta_value(metas, key, |mlist| {
            let mut values = Vec::new();
            for nested in mlist.nested.iter() {
                values.push(expect_literal_string(nested)?);
            }
            Ok(Some(values))
        })
    }

    fn opt_uint(metas: &HashMap<String, Meta>, key: &'static str) -> Result<Option<u64>> {
        with_optional_meta_value(metas, key, |mlist| {
            if mlist.nested.len() != 1 {
                return Err(Error::new(
                    mlist.span(),
                    "expected a single integer value here",
                ));
            }
            expect_literal_uint(mlist.nested.first().unwrap()).map(|u| Some(u))
        })
    }

    fn opt_boolean(metas: &HashMap<String, Meta>, key: &'static str) -> Result<Option<bool>> {
        with_optional_meta_value(metas, key, |mlist| {
            if mlist.nested.len() != 1 {
                return Err(Error::new(
                    mlist.span(),
                    "expected a single boolean value here",
                ));
            }
            expect_literal_boolean(mlist.nested.first().unwrap()).map(|b| Some(b))
        })
    }

    fn boolean(metas: &HashMap<String, Meta>, key: &'static str, default: bool) -> Result<bool> {
        with_meta_value(metas, key, default, |mlist| {
            if mlist.nested.len() != 1 {
                return Err(Error::new(
                    mlist.span(),
                    "expected a single boolean value here",
                ));
            }
            expect_literal_boolean(mlist.nested.first().unwrap())
        })
    }
}

fn with_optional_meta_value<T, F>(
    metas: &HashMap<String, Meta>,
    key: &'static str,
    callback: F,
) -> Result<Option<T>>
where
    F: Fn(&MetaList) -> Result<Option<T>>,
{
    match metas.get(key) {
        None => Ok(None),
        Some(&Meta::List(ref mlist)) => callback(mlist),
        Some(meta) => Err(Error::new(
            meta.span(),
            "expected list-style item here, e.g. `#[option(allow_multiple(true))]`",
        )),
    }
}

fn with_meta_value<T, F>(
    metas: &HashMap<String, Meta>,
    key: &'static str,
    default: T,
    callback: F,
) -> Result<T>
where
    F: Fn(&MetaList) -> Result<T>,
{
    match metas.get(key) {
        None => Ok(default),
        Some(&Meta::List(ref mlist)) => callback(mlist),
        Some(meta) => Err(Error::new(
            meta.span(),
            "expected list-style item here, e.g. `#[option(allow_multiple(true))]`",
        )),
    }
}

fn expect_literal_string(nested: &NestedMeta) -> Result<String> {
    expect_literal(nested, |lit| match lit {
        &Lit::Str(ref s) => Ok(s.value()),
        _ => Err(Error::new(lit.span(), "expected literal string here")),
    })
}

fn expect_literal_boolean(nested: &NestedMeta) -> Result<bool> {
    expect_literal(nested, |lit| match lit {
        &Lit::Bool(ref b) => Ok(b.value),
        _ => Err(Error::new(lit.span(), "expected literal boolean here")),
    })
}

fn expect_literal_uint(nested: &NestedMeta) -> Result<u64> {
    expect_literal(nested, |lit| match lit {
        &Lit::Int(ref i) => {
            let value = i.base10_parse::<u64>()?;
            Ok(value)
        }
        _ => Err(Error::new(lit.span(), "expected literal integer here")),
    })
}

fn expect_literal<T, F>(nested: &NestedMeta, callback: F) -> Result<T>
where
    F: Fn(&Lit) -> Result<T>,
{
    match nested {
        &NestedMeta::Meta(ref meta) => Err(Error::new(meta.span(), "expected literal here")),
        &NestedMeta::Lit(ref lit) => callback(lit),
    }
}
