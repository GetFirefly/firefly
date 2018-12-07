use trackable::*;

use crate::syntax::preprocessor::Preprocessor;
use crate::syntax::tokenizer::Lexer;

fn pp(text: &str) -> Preprocessor<Lexer<&str>> {
    let lexer = Lexer::new(text);
    Preprocessor::new(lexer)
}

#[test]
fn no_directive_works() {
    let src = r#"io:format("Hello")."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["io", ":", "format", "(", r#""Hello""#, ")", "."]
    );
}

#[test]
fn define_works() {
    let src = r#"aaa. -define(foo, [bar, baz]). bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", "."]
    );

    let src = r#"aaa. -define(Foo(A,B), [bar, A, baz, B]). bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", "."]
    );
}

#[test]
fn undef_works() {
    let src = r#"aaa. -undef(foo). bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", "."]
    );
}

#[test]
fn ifdef_works() {
    let src = r#"aaa.-ifdef(foo).bbb.-endif.baz."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "baz", "."]
    );

    let src = r#"-define(foo,bar).aaa.-ifdef(foo).bbb.-endif.baz."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", ".", "baz", "."]
    );
}

#[test]
fn else_works() {
    let src = r#"aaa.-ifdef(foo).bbb.-else.ccc.-endif.baz."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "ccc", ".", "baz", "."]
    );

    let src = r#"-define(foo,bar).aaa.-ifdef(foo).bbb.-else.ccc.-endif.baz."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", ".", "baz", "."]
    );
}

#[test]
fn ifndef_works() {
    let src = r#"aaa.-ifndef(foo).bbb.-endif.baz."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", ".", "baz", "."]
    );

    let src = r#"-define(foo,bar).aaa.-ifndef(foo).bbb.-endif.baz."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "baz", "."]
    );
}

#[test]
fn error_and_warning_works() {
    let src = r#"aaa. -error("foo"). bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", "."]
    );

    let src = r#"aaa. -warning("foo"). bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bbb", "."]
    );
}

#[test]
fn include_works() {
    let src = r#"foo.-include("tests/testdata/preprocessor/bar.hrl").baz."#;
    let tokens = track_try_unwrap!(pp(src).collect::<Result<Vec<_>, _>>());

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["foo", ".", "bar", ".", "baz", "."]
    );
}

#[test]
fn include_lib_works() {
    let src = r#"foo.-include_lib("tests/testdata/preprocessor/bar.hrl").baz."#;
    let tokens = track_try_unwrap!(pp(src).collect::<Result<Vec<_>, _>>());

    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["foo", ".", "bar", ".", "baz", "."]
    );
}

#[test]
fn macro_expansion_works() {
    let src = r#"-define(foo,bar).aaa.?foo.bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "bar", ".", "bbb", "."]
    );

    let src = r#"-define(foo(A), {bar, A}).aaa.?foo([1,2]).bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "{", "bar", ",", "[", "1", ",", "2", "]", "}", ".", "bbb", ".",]
    );

    let src = r#"-define(foo(A), {bar, ??A}).aaa.?foo([1,2]).bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        [
            "aaa",
            ".",
            "{",
            "bar",
            ",",
            r#""[1,2]""#,
            "}",
            ".",
            "bbb",
            ".",
        ]
    );

    let src = r#"-define(foo, {bar, ?LINE}). ?foo."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["{", "bar", ",", "1", "}", "."]
    );

    let src = r#"-define(foo(A), {A, ??A}). -define(bar, baz). ?foo(?bar)."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["{", "baz", ",", r#""?bar""#, "}", "."]
    );

    let src = r#"-define(foo(A), ?bar(A)). -define(bar(A), A). ?foo(baz)."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["baz", "."]
    );
}

#[test]
fn predefined_macro_works() {
    let src = r#"aaa.?LINE.bbb."#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["aaa", ".", "1", ".", "bbb", "."]
    );
}

#[test]
fn args_for_expanded_tokens_test() {
    let src = r#"
-define(yo, -module).
?yo(prog).
"#;
    let tokens = pp(src).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
        ["-", "module", "(", "prog", ")", "."]
    );
}
