/// Used in the grammar for easy span creation
macro_rules! span {
    ($l:expr, $r:expr) => {
        SourceSpan::new($l, $r)
    };
    ($i:expr) => {
        SourceSpan::new($i, $i)
    };
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unknown_lints)]
#[allow(clippy)]
#[allow(unused_parens)]
pub(crate) mod grammar {
    // During the build step, `build.rs` will output the generated parser to `OUT_DIR` to avoid
    // adding it to the source directory, so we just directly include the generated parser here.
    //
    // Even with `.gitignore` and the `exclude` in the `Cargo.toml`, the generated parser can still
    // end up in the source directory. This could happen when `cargo build` builds the file out of
    // the Cargo cache (`$HOME/.cargo/registrysrc`), and the build script would then put its output
    // in that cached source directory because of https://github.com/lalrpop/lalrpop/issues/280.
    // Later runs of `cargo vendor` then copy the source from that directory, including the
    // generated file.
    include!(concat!(env!("OUT_DIR"), "/parser/grammar.rs"));
}

mod errors;

use std::sync::Arc;

use firefly_parser::{Parse as GParse, Parser as GParser};
use firefly_parser::{Scanner, Source};
use firefly_util::diagnostics::{CodeMap, DiagnosticsHandler, SourceIndex};

pub use self::errors::ParserError;

use crate::ast;
use crate::lexer::{Lexer, Token};

pub type Parser = GParser<()>;
pub trait Parse<T> = GParse<T, Config = (), Error = ParserError>;

impl GParse for ast::Ast {
    type Parser = grammar::AbstractFormatParser;
    type Error = ParserError;
    type Config = ();
    type Token = Result<(SourceIndex, Token, SourceIndex), ParserError>;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParserError::RootFile { source, path }
    }

    fn parse<S>(
        parser: &GParser<Self::Config>,
        diagnostics: &DiagnosticsHandler,
        source: S,
    ) -> Result<Self, Self::Error>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        Self::parse_tokens(diagnostics, parser.codemap.clone(), lexer)
    }

    fn parse_tokens<S>(
        diagnostics: &DiagnosticsHandler,
        codemap: Arc<CodeMap>,
        tokens: S,
    ) -> Result<Self, Self::Error>
    where
        S: IntoIterator<Item = Self::Token>,
    {
        let result = Self::Parser::new().parse(diagnostics, &codemap, tokens);
        match result {
            Ok(ast) => Ok(ast),
            Err(lalrpop_util::ParseError::User { error }) => Err(error.into()),
            Err(err) => Err(ParserError::from(err).into()),
        }
    }
}

impl GParse for ast::Root {
    type Parser = grammar::RootParser;
    type Error = ParserError;
    type Config = ();
    type Token = Result<(SourceIndex, Token, SourceIndex), ParserError>;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParserError::RootFile { source, path }
    }

    fn parse<S>(
        parser: &GParser<Self::Config>,
        diagnostics: &DiagnosticsHandler,
        source: S,
    ) -> Result<Self, Self::Error>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        Self::parse_tokens(diagnostics, parser.codemap.clone(), lexer)
    }

    fn parse_tokens<S>(
        diagnostics: &DiagnosticsHandler,
        codemap: Arc<CodeMap>,
        tokens: S,
    ) -> Result<Self, Self::Error>
    where
        S: IntoIterator<Item = Self::Token>,
    {
        let result = Self::Parser::new().parse(diagnostics, &codemap, tokens);
        match result {
            Ok(root) => Ok(root),
            Err(lalrpop_util::ParseError::User { error }) => Err(error.into()),
            Err(err) => Err(ParserError::from(err).into()),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use firefly_util::diagnostics::*;

    use super::*;
    use crate::ast::*;

    fn parse<T, S>(codemap: Arc<CodeMap>, input: S) -> T
    where
        T: Parse<T, Config = (), Error = ParserError>,
        S: AsRef<str>,
    {
        let emitter = Arc::new(DefaultEmitter::new(ColorChoice::Auto));
        let diagnostics =
            DiagnosticsHandler::new(DiagnosticsConfig::default(), codemap.clone(), emitter);
        let parser = Parser::new((), codemap);
        match parser.parse_string::<T, S, ParserError>(&diagnostics, input) {
            Ok(ast) => return ast,
            Err(err) => {
                diagnostics.error(err);
                panic!("parse failed");
            }
        }
    }

    const SIMPLE: &'static str = "{application, simple, []}.";
    const RICH: &'static str = r#"
%% A multi-line
%% comment
{application, example,
  [{description, "An example application"},
   {vsn, "0.1.0-rc0"},
   %% Another comment
   {modules, [example_app, example_sup, example_worker]},
   {registered, [example_registry]},
   {applications, [kernel, stdlib, sasl]},
   {mod, {example_app, []}}
  ]}.
"#;

    #[test]
    fn simple_app_resource_test() {
        let codemap = Arc::new(CodeMap::new());
        let _: Root = parse(codemap.clone(), SIMPLE);
    }

    #[test]
    fn rich_app_resource_test() {
        let codemap = Arc::new(CodeMap::new());
        let _: Root = parse(codemap.clone(), RICH);
    }

    #[test]
    fn complex_ast() {
        let codemap = Arc::new(CodeMap::new());
        let _: Ast = parse(
            codemap.clone(),
            r#"
{attribute,{1,1},file,{"library/beam/tests/testdata/ast/test.erl",1}}.
 {attribute,{1,2},module,test}.
 {attribute,{3,2},compile,debug_info}.
 {attribute,{4,2},compile,[{foo,#{bar => true}}]}.
 {attribute,{5,2},foo_attribute,bar}.
 {attribute,{7,2},behaviour,test}.
 {attribute,{8,2},behavior,test2}.
 {attribute,{10,2},export,[{literals,0}]}.
 {attribute,{11,2},export,[{hello,1}]}.
 {attribute,{12,2},export,[{map_fun,2}]}.
 {attribute,{13,2},export,[{cons,2}]}.
 {attribute,{14,2},export,[{to_my_list,1}]}.
 {attribute,{15,2},export,[{my_record,0}]}.
 {attribute,{16,2},export,[{guard,1}]}.
 {attribute,{17,2},export,[{sum,1},{op,1}]}.
 {attribute,{19,2},export_type,[{my_list,1}]}.
 {attribute,{20,2},export_type,[{my_cons,2}]}.
 {attribute,{22,2},import,{lists,[{usort,1}]}}.
 {attribute,{24,2},on_load,{nif_impl,0}}.
 {attribute,{25,2},nifs,[{nif_impl,0}]}.
 {attribute,
     {27,2},
     callback,
     {{hello,1},
      [{type,
           {27,16},
           'fun',
           [{type,
                {27,16},
                product,
                [{ann_type,
                     {27,17},
                     [{var,{27,17},'Name'},{type,{27,25},binary,[]}]}]},
            {type,
                {27,38},
                union,
                [{atom,{27,38},ok},
                 {type,
                     {27,43},
                     tuple,
                     [{atom,{27,44},error},
                      {ann_type,
                          {27,51},
                          [{var,{27,51},'Reason'},
                           {type,{27,61},term,[]}]}]}]}]}]}}.
 {attribute,{28,2},optional_callbacks,[{hello,1}]}.
 {attribute,
     {30,2},
     opaque,
     {my_list,
         {type,
             {30,23},
             union,
             [{user_type,
                  {30,23},
                  my_cons,
                  [{var,{30,31},'E'},
                   {user_type,{30,34},my_list,[{var,{30,42},'E'}]}]},
              {atom,{30,48},nil}]},
         [{var,{30,17},'E'}]}}.
 {attribute,
     {31,2},
     type,
     {my_cons,
         {type,{31,24},tuple,[{var,{31,25},'H'},{var,{31,28},'T'}]},
         [{var,{31,15},'H'},{var,{31,18},'T'}]}}.
 {attribute,
     {33,2},
     type,
     {bits0,{type,{33,18},binary,[{integer,33,0},{integer,33,0}]},[]}}.
 {attribute,
     {34,2},
     type,
     {bits1,{type,{34,18},binary,[{integer,{34,22},8},{integer,34,0}]},[]}}.
 {attribute,
     {35,2},
     type,
     {bits2,{type,{35,18},binary,[{integer,35,0},{integer,{35,24},8}]},[]}}.
 {attribute,
     {36,2},
     type,
     {bits3,
         {type,{36,18},binary,[{integer,{36,22},8},{integer,{36,29},1}]},
         []}}.
 {attribute,{38,2},type,{foo,{type,{38,16},tuple,any},[]}}.
 {attribute,
     {39,2},
     type,
     {lists,{type,{39,18},list,[{type,{39,23},any,[]}]},[]}}.
 {attribute,
     {41,2},
     type,
     {external_type_alias,
         {remote_type,
             {41,32},
             [{atom,{41,32},unicode},{atom,{41,40},chardata},[]]},
         []}}.
 {attribute,
     {43,2},
     type,
     {unary_op_type,{op,{43,26},'-',{integer,{43,27},100}},[]}}.
 {attribute,
     {44,2},
     type,
     {binary_op_type,
         {op,{44,31},'*',{integer,{44,27},100},{integer,{44,33},8}},
         []}}.
 {attribute,{46,2},deprecated,{hello,1}}.
 {attribute,{47,2},deprecated,{map_fun,2,next_major_release}}.
 {warning,{{49,2},epp,{warning,"this is a warning"}}}.
 {attribute,
     {55,2},
     record,
     {my_record,
         [{record_field,{57,11},{atom,{57,11},a}},
          {typed_record_field,
              {record_field,{58,11},{atom,{58,11},b},{integer,{58,15},10}},
              {type,{58,21},integer,[]}},
          {typed_record_field,
              {record_field,{59,11},{atom,{59,11},c}},
              {type,{59,16},pid,[]}},
          {record_field,{60,11},{atom,{60,11},d},{atom,{60,15},foo}}]}}.
 {attribute,
     {63,2},
     type,
     {record_ty,{type,{63,22},record,[{atom,{63,23},my_record}]},[]}}.
 {attribute,
     {65,2},
     spec,
     {{literals,0},
      [{type,
           {65,15},
           'fun',
           [{type,{65,15},product,[]},
            {type,
                {65,21},
                tuple,
                [{type,{65,22},integer,[]},
                 {type,{65,33},neg_integer,[]},
                 {type,{65,48},float,[]},
                 {type,{65,57},atom,[]},
                 {type,{65,65},list,[]},
                 {type,{65,73},binary,[]},
                 {type,{65,83},bitstring,[]},
                 {type,{65,96},map,any},
                 {type,{65,103},pid,[]},
                 {type,{65,110},reference,[]}]}]}]}}.
 {function,
     {66,1},
     literals,0,
     [{clause,
          {66,1},
          [],[],
          [{tuple,
               {67,5},
               [{integer,{68,7},123},
                {op,{69,7},'-',{integer,{69,8},123}},
                {float,{70,7},12.3},
                {atom,{71,7},foo},
                {cons,
                    {72,7},
                    {integer,{72,8},1},
                    {cons,
                        {72,10},
                        {integer,{72,10},2},
                        {cons,{72,12},{integer,{72,12},3},{nil,{72,13}}}}},
                {bin,
                    {73,7},
                    [{bin_element,
                         {73,9},
                         {string,{73,9},"123"},
                         default,default}]},
                {bin,
                    {74,7},
                    [{bin_element,
                         {74,9},
                         {string,{74,9},"123"},
                         default,default},
                     {bin_element,
                         {74,16},
                         {integer,{74,16},2},
                         {integer,{74,18},2},
                         default}]},
                {map,
                    {75,7},
                    [{map_field_assoc,
                         {75,13},
                         {integer,{75,9},123},
                         {atom,{75,16},abc}}]},
                {call,{76,7},{atom,{76,7},self},[]},
                {call,{77,7},{atom,{77,7},make_ref},[]}]}]}]}.
 {function,
     {80,1},
     nif_impl,0,
     [{clause,
          {80,1},
          [],[],
          [{call,
               {81,5},
               {remote,{81,11},{atom,{81,5},erlang},{atom,{81,12},nif_error}},
               [{atom,{81,22},failed}]}]}]}.
 {function,
     {83,1},
     hello,1,
     [{clause,
          {83,1},
          [{bin,
               {83,7},
               [{bin_element,{83,9},{var,{83,9},'Name'},default,[binary]}]}],
          [],
          [{call,
               {84,5},
               {remote,{84,7},{atom,{84,5},io},{atom,{84,8},format}},
               [{string,{84,15},"Hello ~s\n"},
                {cons,{84,29},{var,{84,30},'Name'},{nil,{84,34}}}]},
           {atom,{85,5},ok}]}]}.
 {attribute,
     {87,2},
     spec,
     {{map_fun,2},
      [{type,
           {87,14},
           bounded_fun,
           [{type,
                {87,14},
                'fun',
                [{type,
                     {87,14},
                     product,
                     [{var,{87,15},'Fun'},{var,{87,20},'List'}]},
                 {var,{87,29},'Result'}]},
            [{type,
                 {88,7},
                 constraint,
                 [{atom,{88,7},is_subtype},
                  [{var,{88,7},'Fun'},
                   {type,
                       {88,19},
                       'fun',
                       [{type,{88,19},product,[{var,{88,20},'Input'}]},
                        {var,{88,30},'Result'}]}]]},
             {type,
                 {89,7},
                 constraint,
                 [{atom,{89,7},is_subtype},
                  [{var,{89,7},'Input'},{type,{89,16},term,[]}]]},
             {type,
                 {90,7},
                 constraint,
                 [{atom,{90,7},is_subtype},
                  [{var,{90,7},'List'},
                   {type,{90,15},list,[{var,{90,16},'Input'}]}]]},
             {type,
                 {91,7},
                 constraint,
                 [{atom,{91,7},is_subtype},
                  [{var,{91,7},'Result'},{type,{91,17},term,[]}]]}]]}]}}.
 {function,
     {92,1},
     map_fun,2,
     [{clause,
          {92,1},
          [{var,{92,9},'Fun'},{var,{92,14},'List'}],
          [],
          [{lc,{93,5},
               {call,{93,6},{var,{93,6},'Fun'},[{var,{93,10},'X'}]},
               [{generate,
                    {93,18},
                    {var,{93,16},'X'},
                    {var,{93,21},'List'}}]}]}]}.
 {attribute,
     {95,2},
     spec,
     {{cons,2},
      [{type,
           {95,11},
           bounded_fun,
           [{type,
                {95,11},
                'fun',
                [{type,{95,11},product,[{var,{95,12},'H'},{var,{95,15},'T'}]},
                 {user_type,
                     {95,21},
                     my_cons,
                     [{var,{95,29},'H'},{var,{95,32},'T'}]}]},
            [{type,
                 {96,7},
                 constraint,
                 [{atom,{96,7},is_subtype},
                  [{var,{96,7},'H'},{type,{96,12},term,[]}]]},
             {type,
                 {97,7},
                 constraint,
                 [{atom,{97,7},is_subtype},
                  [{var,{97,7},'T'},{type,{97,12},term,[]}]]}]]}]}}.
 {function,
     {98,1},
     cons,2,
     [{clause,
          {98,1},
          [{var,{98,6},'H'},{var,{98,9},'T'}],
          [],
          [{tuple,{99,5},[{var,{99,6},'H'},{var,{99,9},'T'}]}]}]}.
 {attribute,
     {101,2},
     spec,
     {{to_my_list,1},
      [{type,
           {101,17},
           'fun',
           [{type,
                {101,17},
                product,
                [{type,{101,18},list,[{var,{101,19},'E'}]}]},
            {user_type,{101,26},my_list,[{var,{101,34},'E'}]}]}]}}.
 {function,
     {102,1},
     to_my_list,1,
     [{clause,{102,1},[{nil,{102,12}}],[],[{atom,{102,24},nil}]},
      {clause,
          {103,1},
          [{cons,{103,12},{var,{103,13},'H'},{var,{103,17},'T'}}],
          [],
          [{call,
               {103,24},
               {atom,{103,24},cons},
               [{var,{103,29},'H'},
                {call,
                    {103,32},
                    {atom,{103,32},to_my_list},
                    [{var,{103,43},'T'}]}]}]}]}.
 {attribute,
     {105,2},
     spec,
     {{my_record,0},
      [{type,
           {105,16},
           'fun',
           [{type,{105,16},product,[]},
            {type,
                {105,22},
                record,
                [{atom,{105,23},my_record},
                 {type,
                     {105,33},
                     field_type,
                     [{atom,{105,33},c},{type,{105,38},pid,[]}]}]}]}]}}.
 {function,
     {106,1},
     my_record,0,
     [{clause,
          {106,1},
          [],[],
          [{block,
               {107,5},
               [{match,
                    {108,9},
                    {var,{108,9},'Rec'},
                    {record,
                        {108,15},
                        my_record,
                        [{record_field,
                             {109,12},
                             {atom,{109,12},c},
                             {call,{109,16},{atom,{109,16},self},[]}},
                         {record_field,
                             {110,12},
                             {var,{110,12},'_'},
                             {atom,{110,16},'_'}}]}},
                {match,
                    {112,9},
                    {var,{112,9},'_Index'},
                    {record_index,{112,18},my_record,{atom,{112,29},c}}},
                {match,
                    {113,9},
                    {var,{113,9},'_Access'},
                    {record_field,
                        {113,22},
                        {var,{113,19},'Rec'},
                        my_record,
                        {atom,{113,33},c}}},
                {var,{114,9},'Rec'}]}]}]}.
 {attribute,
     {117,2},
     spec,
     {{guard,1},
      [{type,
           {117,12},
           'fun',
           [{type,
                {117,12},
                product,
                [{type,
                     {117,13},
                     union,
                     [{type,{117,13},integer,[]},{type,{117,25},atom,[]}]}]},
            {type,
                {117,36},
                union,
                [{type,{117,36},integer,[]},{type,{117,48},atom,[]}]}]},
       {type,
           {118,12},
           'fun',
           [{type,
                {118,12},
                product,
                [{type,
                     {118,13},
                     range,
                     [{integer,{118,13},1},{integer,{118,16},99}]}]},
            {type,{118,23},float,[]}]},
       {type,
           {119,12},
           'fun',
           [{type,{119,12},product,[{type,{119,13},map,any}]},
            {type,{119,23},term,[]}]},
       {type,
           {120,12},
           'fun',
           [{type,
                {120,12},
                product,
                [{type,
                     {120,13},
                     tuple,
                     [{type,{120,14},term,[]},
                      {type,{120,22},map,any},
                      {type,{120,29},binary,[]}]}]},
            {type,{120,43},binary,[]}]},
       {type,
           {121,12},
           'fun',
           [{type,{121,12},product,[{type,{121,13},tuple,any}]},
            {type,{121,25},non_neg_integer,[]}]}]}}.
 {function,
     {122,1},
     guard,1,
     [{clause,
          {122,1},
          [{var,{122,7},'X'}],
          [[{call,{122,15},{atom,{122,15},is_integer},[{var,{122,26},'X'}]}],
           [{call,{122,30},{atom,{122,30},is_atom},[{var,{122,38},'X'}]}]],
          [{var,{122,44},'X'}]},
      {clause,
          {123,1},
          [{var,{123,7},'X'}],
          [[{call,{123,15},{atom,{123,15},is_integer},[{var,{123,26},'X'}]},
            {op,{123,32},'<',{integer,{123,30},0},{var,{123,34},'X'}},
            {op,{123,39},'<',{var,{123,37},'X'},{integer,{123,41},100}}]],
          [{op,{123,51},'/',{integer,{123,48},10},{var,{123,53},'X'}}]},
      {clause,
          {124,1},
          [{map,
               {124,7},
               [{map_field_exact,
                    {124,15},
                    {atom,{124,9},hello},
                    {var,{124,18},'X'}}]}],
          [[{op,{124,38},
                'orelse',
                {call,{124,27},{atom,{124,27},is_atom},[{var,{124,35},'X'}]},
                {op,{124,60},
                    'andalso',
                    {call,
                        {124,46},
                        {atom,{124,46},is_integer},
                        [{var,{124,57},'X'}]},
                    {op,{124,70},
                        '<',
                        {var,{124,68},'X'},
                        {integer,{124,72},0}}}}]],
          [{var,{124,77},'X'}]},
      {clause,
          {125,1},
          [{tuple,
               {125,7},
               [{var,{125,8},'_'},
                {map,{125,11},[]},
                {bin,
                    {125,16},
                    [{bin_element,
                         {125,18},
                         {integer,{125,18},10},
                         default,default},
                     {bin_element,
                         {125,22},
                         {var,{125,22},'Bin'},
                         default,
                         [binary]}]}]}],
          [],
          [{var,{125,41},'Bin'}]},
      {clause,
          {126,1},
          [{var,{126,7},'X'}],
          [[{call,{126,15},{atom,{126,15},is_tuple},[{var,{126,24},'X'}]}]],
          [{call,{126,30},{atom,{126,30},tuple_size},[{var,{126,41},'X'}]}]}]}.
 {attribute,
     {128,2},
     spec,
     {{sum,1},
      [{type,
           {128,10},
           'fun',
           [{type,
                {128,10},
                product,
                [{type,{128,11},list,[{type,{128,12},number,[]}]}]},
            {type,{128,26},number,[]}]}]}}.
 {function,
     {129,1},
     sum,1,
     [{clause,
          {129,1},
          [{var,{129,5},'List'}],
          [],
          [{call,
               {130,6},
               {named_fun,
                   {130,6},
                   'Rec',
                   [{clause,
                        {130,10},
                        [{nil,{130,15}}],
                        [],
                        [{integer,{130,22},0}]},
                    {clause,
                        {131,10},
                        [{cons,
                             {131,15},
                             {var,{131,16},'X'},
                             {var,{131,20},'Xs'}}],
                        [],
                        [{op,{131,30},
                             '+',
                             {var,{131,28},'X'},
                             {call,
                                 {131,32},
                                 {var,{131,32},'Rec'},
                                 [{var,{131,36},'Xs'}]}}]}]},
               [{var,{132,11},'List'}]}]}]}.
 {attribute,
     {134,2},
     spec,
     {{op,1},
      [{type,
           {134,9},
           'fun',
           [{type,{134,9},product,[{type,{134,10},integer,[]}]},
            {type,{134,24},integer,[]}]}]}}.
 {function,
     {135,1},
     op,1,
     [{clause,
          {135,1},
          [{var,{135,4},'Num'}],
          [],
          [{op,{136,15},
               'band',
               {op,{136,10},'+',{var,{136,6},'Num'},{integer,{136,12},1}},
               {integer,{136,20},4294967295}}]}]}.
 {function,
     {138,1},
     catcher,1,
     [{clause,
          {138,1},
          [{var,{138,9},'Arg'}],
          [],
          [{'catch',
               {139,5},
               {call,
                   {139,11},
                   {atom,{139,11},throw},
                   [{var,{139,17},'Arg'}]}}]}]}.
 {function,
     {141,1},
     tryer,1,
     [{clause,
          {141,1},
          [{var,{141,7},'Arg'}],
          [],
          [{'try',
               {142,5},
               [{var,{143,9},'Arg'}],
               [{clause,{145,9},[{atom,{145,9},foo}],[],[{atom,{146,13},ok}]}],
               [{clause,
                    {148,9},
                    [{tuple,
                         {148,9},
                         [{var,{148,9},'_'},
                          {var,{148,11},'_'},
                          {var,{148,13},'_'}]}],
                    [],
                    [{atom,{149,13},error}]}],
               [{atom,{151,9},ok}]}]}]}.
 {function,
     {154,1},
     conditional,1,
     [{clause,
          {154,1},
          [{var,{154,13},'Arg'}],
          [],
          [{'if',
               {155,5},
               [{clause,
                    {156,9},
                    [],
                    [[{call,
                          {156,9},
                          {atom,{156,9},is_list},
                          [{var,{156,17},'Arg'}]}]],
                    [{call,
                         {157,13},
                         {atom,{157,13},tl},
                         [{var,{157,16},'Arg'}]}]},
                {clause,
                    {159,9},
                    [],
                    [[{call,
                          {159,9},
                          {atom,{159,9},is_map},
                          [{var,{159,16},'Arg'}]}]],
                    [{call,
                         {160,13},
                         {atom,{160,13},map_get},
                         [{atom,{160,21},foo},{var,{160,26},'Arg'}]}]},
                {clause,
                    {162,9},
                    [],
                    [[{atom,{162,9},true}]],
                    [{atom,{163,13},error}]}]}]}]}.
 {function,
     {166,1},
     matcher,1,
     [{clause,
          {166,1},
          [{var,{166,9},'Arg'}],
          [],
          [{'case',
               {167,5},
               {var,{167,10},'Arg'},
               [{clause,
                    {168,9},
                    [{cons,{168,9},{var,{168,10},'_'},{var,{168,14},'_'}}],
                    [],
                    [{atom,{168,20},list}]},
                {clause,{169,9},[{map,{169,9},[]}],[],[{atom,{169,16},map}]},
                {clause,
                    {170,9},
                    [{var,{170,9},'A'}],
                    [[{call,
                          {170,16},
                          {atom,{170,16},is_number},
                          [{var,{170,26},'A'}]}]],
                    [{atom,{170,32},number}]},
                {clause,
                    {171,9},
                    [{var,{171,9},'_'}],
                    [],
                    [{atom,{171,14},unknown}]}]}]}]}.
 {function,
     {174,1},
     receive_impatient,1,
     [{clause,
          {174,1},
          [{var,{174,19},'Arg'}],
          [],
          [{'receive',
               {175,5},
               [{clause,
                    {176,9},
                    [{tuple,
                         {176,9},
                         [{var,{176,10},'Sender'},{atom,{176,18},ping}]}],
                    [],
                    [{op,{177,20},
                         '!',
                         {var,{177,13},'Sender'},
                         {atom,{177,22},pong}}]}],
               {integer,{179,11},1000},
               [{atom,{180,9},failed}]}]}]}.
 {function,
     {183,1},
     receive_patient,1,
     [{clause,
          {183,1},
          [{var,{183,17},'Arg'}],
          [],
          [{'receive',
               {184,5},
               [{clause,
                    {185,9},
                    [{tuple,
                         {185,9},
                         [{var,{185,10},'Sender'},{atom,{185,18},ping}]}],
                    [],
                    [{op,{186,20},
                         '!',
                         {var,{186,13},'Sender'},
                         {atom,{186,22},pong}}]}]}]}]}.
 {function,
     {189,1},
     captures,0,
     [{clause,
          {189,1},
          [],[],
          [{match,
               {190,5},
               {var,{190,5},'Internal'},
               {'fun',{190,16},{function,matcher,1}}},
           {match,
               {191,5},
               {var,{191,5},'External'},
               {'fun',
                   {191,16},
                   {function,
                       {atom,{191,20},erlang},
                       {atom,{191,27},display},
                       {integer,{191,35},1}}}},
           {tuple,
               {192,5},
               [{var,{192,6},'Internal'},{var,{192,16},'External'}]}]}]}.
 {eof,{194,1}}.
"#,
        );
    }
}
