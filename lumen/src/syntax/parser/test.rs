use crate::syntax::parser::cst::{Expr, Form, Pattern, Type};
use crate::syntax::parser::{Parser, TokenReader};
use crate::syntax::preprocessor::Preprocessor;
use crate::syntax::tokenizer::{Lexer, PositionRange};

use trackable::*;

macro_rules! parse_expr {
    ($text:expr) => {
        let mut parser = Parser::new(TokenReader::new(Preprocessor::new(Lexer::new($text))));
        let value: Expr = track_try_unwrap!(parser.parse(), "text={:?}", $text);
        assert_eq!(value.end_position().offset(), $text.len());
    };
}

macro_rules! parse_pattern {
    ($text:expr) => {
        let mut parser = Parser::new(TokenReader::new(Preprocessor::new(Lexer::new($text))));
        let value: Pattern = track_try_unwrap!(parser.parse(), "text={:?}", $text);
        assert_eq!(value.end_position().offset(), $text.len());
    };
}

macro_rules! parse_type {
    ($text:expr) => {
        let mut parser = Parser::new(TokenReader::new(Preprocessor::new(Lexer::new($text))));
        let value: Type = track_try_unwrap!(parser.parse(), "text={:?}", $text);
        assert_eq!(value.end_position().offset(), $text.len());
    };
}

macro_rules! parse_form {
    ($text:expr) => {
        let mut parser = Parser::new(TokenReader::new(Preprocessor::new(Lexer::new($text))));
        let value: Form = track_try_unwrap!(parser.parse(), "text={:?}", $text);
        assert_eq!(value.end_position().offset(), $text.len());
    };
}

#[test]
fn parse_expr_works() {
    // literals
    parse_expr!("foo");
    parse_expr!("$c");
    parse_expr!("1.2");
    parse_expr!("123");
    parse_expr!(r#""foo""#);
    parse_expr!(r#""foo" "bar" "baz""#);

    // variable
    parse_expr!("Foo");

    // tuple
    parse_expr!("{}");
    parse_expr!("{1}");
    parse_expr!("{1, 2, 3}");

    // map
    parse_expr!("#{}");
    parse_expr!("#{a => b}");
    parse_expr!("#{a := b}");
    parse_expr!("#{a => b, 1 := 2}");

    // map update
    parse_expr!("M#{}");
    parse_expr!("(#{a => 10})#{a := b, 1 => 2}");

    // record
    parse_expr!("#foo{}");
    parse_expr!("#foo{a = b}");
    parse_expr!("#foo{a = b, _ = 10}");

    // record update
    parse_expr!("R#foo{bar = 10}");
    parse_expr!("(#foo{})#foo{bar = 10, baz = 20}");

    // record field access
    parse_expr!("R#foo.bar");
    parse_expr!("(#foo{})#foo.bar");

    // record field index
    parse_expr!("#foo.bar");

    // proper list
    parse_expr!("[]");
    parse_expr!("[1]");
    parse_expr!("[1, 2, 3]");

    // improper list
    parse_expr!("[1 | 2]");
    parse_expr!("[1, 2 | 3]");

    // list comprehension
    parse_expr!("[x || _ <- [1,2,3]]");
    parse_expr!("[x || X <- [1,2,3], filter(X), _ <= <<1,2,3>>]");

    // bitstring
    parse_expr!("<<>>");
    parse_expr!("<<10>>");
    parse_expr!("<<1, 2, 3>>");
    parse_expr!("<<100:2>>");
    parse_expr!("<<1/little>>");
    parse_expr!("<<1:2/little-unit:8>>");

    // bitstring comprehension
    parse_expr!("<< <<x>> || _ <- [1,2,3]>>");
    parse_expr!("<< <<x>> || X <- [1,2,3], filter(X), _ <= <<1,2,3>> >>");

    // parenthesized
    parse_expr!("( 1 )");

    // local call
    parse_expr!("foo()");
    parse_expr!("Foo(1)");
    parse_expr!(r#"(list_to_atom("foo"))(1, 2, [3])"#);

    // remote call
    parse_expr!("foo:bar()");
    parse_expr!("Foo:Bar(1)");
    parse_expr!(r#"(list_to_atom("foo")):bar(1, 2, [3])"#);

    // local fun
    parse_expr!("fun foo/2");

    // remote fun
    parse_expr!("fun foo:bar/2");
    parse_expr!("fun Foo:Bar/Baz");

    // anonymous fun
    parse_expr!("fun () -> ok end");
    parse_expr!("fun (a) -> ok; (B) -> err end");
    parse_expr!("fun (a) when true -> ok; (B) -> err end");

    // named fun
    parse_expr!("fun Foo() -> ok end");
    parse_expr!("fun Foo(a) -> ok; Foo(B) -> err end");
    parse_expr!("fun Foo(a) when true -> Foo(b); Foo(B) -> err end");

    // unary op
    parse_expr!("+10");
    parse_expr!("-20");
    parse_expr!("not false");
    parse_expr!("bnot Foo");

    // binary op
    parse_expr!("1 =:= 2");
    parse_expr!("Pid ! [1, 2] ++ [3] -- [1]");
    parse_expr!("foo() ++ bar()");

    // match
    parse_expr!("1 = 2");
    parse_expr!("1 = 2 = 3");
    parse_expr!("[A, 2, {}] = [10 | B]");

    // block
    parse_expr!("begin 1, 2, 3 end");

    // catch
    parse_expr!("catch [1,2,3]");

    // if
    parse_expr!("if true -> 1, 2, 3 end");
    parse_expr!("if true; false, true -> 1, 2, 3 end");
    parse_expr!("if true -> 1; false -> 2; _ -> ok end");

    // case
    parse_expr!("case 1 of 2 -> 3 end");
    parse_expr!("case 1 of 2 -> 3; _ -> ok end");
    parse_expr!("case 1 of A when A == 2; true; 1, 2, false -> 3 end");

    // receive
    parse_expr!("receive foo -> bar end");
    parse_expr!("receive Foo when Foo == 2 -> bar; 1 -> 2 end");
    parse_expr!("receive Foo when Foo == 2 -> bar; 1 -> 2 after 10 * 2 -> foo(), done end");

    // try
    parse_expr!("try foo, bar catch baz -> 1; _ -> qux end");
    parse_expr!("try foo, bar of 10 -> 2; _ -> 30 catch baz -> 1; _:_ -> qux end");
    parse_expr!("try foo, bar after baz, qux end");
    parse_expr!("try foo of _ -> 1 catch throw:_ -> ok end");

    parse_expr!("try foo of _ -> 1 after ok end");
    parse_expr!("try foo of _ -> 1 catch _ -> err after ok end");
}

#[test]
fn parse_pattern_works() {
    // literals
    parse_pattern!("foo");
    parse_pattern!("$c");
    parse_pattern!("1.2");
    parse_pattern!("123");
    parse_pattern!(r#""foo""#);

    // variable
    parse_pattern!("Foo");

    // bitstring
    parse_pattern!("<<>>");
    parse_pattern!("<<10>>");
    parse_pattern!("<<1, 2, 3>>");
    parse_pattern!("<<100:2>>");
    parse_pattern!("<<1/little>>");
    parse_pattern!("<<1:2/little-unit:8>>");

    // proper list
    parse_pattern!("[]");
    parse_pattern!("[1]");
    parse_pattern!("[1, 2, 3]");

    // improper list
    parse_pattern!("[1 | 2]");
    parse_pattern!("[1, 2 | 3]");

    // map
    parse_pattern!("#{}");
    parse_pattern!("#{a := b}");
    parse_pattern!("#{a := B, 1 := 2}");

    // tuple
    parse_pattern!("{}");
    parse_pattern!("{1}");
    parse_pattern!("{1, 2, 3}");

    // unary op
    parse_pattern!("+10");
    parse_pattern!("-20");

    // binary op
    parse_pattern!("[1] ++ [2,3]");

    // parenthesized
    parse_pattern!("( [1,2,3] )");

    // record
    parse_pattern!("#foo{}");
    parse_pattern!("#foo{a = b}");
    parse_pattern!("#foo{a = b, _ = 10}");

    // record field index
    parse_pattern!("#foo.bar");

    // match
    parse_pattern!("{A, B = 2, 3} = {1, 2, 3}");
}

#[test]
fn parse_type_works() {
    // integer
    parse_type!("10");
    parse_type!("-10");
    parse_type!("(3)");
    parse_type!("(10 - 2)");
    parse_type!("1 + 2 - 3 rem 4");
    parse_type!("(1 + 2) - 3");
    parse_type!("1 + (2 - -3)");

    // integer range
    parse_type!("0..10");
    parse_type!("-10..+10");
    parse_type!("(1 + 2)..(10 * 30 - 1)");

    // annotated
    parse_type!("A :: 10");

    // list
    parse_type!("[]");
    parse_type!("[foo]");
    parse_type!("[foo, ...]");

    // parenthesized
    parse_type!("([10])");

    // tuple
    parse_type!("{1, 2, 3}");

    // map
    parse_type!("#{a => 10, b := 20}");

    // record
    parse_type!("#foo{bar = integer()}");

    // bitstring
    parse_type!("<<>>");
    parse_type!("<<_:1>>");
    parse_type!("<<_:_*3>>");
    parse_type!("<<_:10,_:_*3>>");

    // call
    parse_type!("foo()");
    parse_type!("foo:bar(1,2,3)");

    // fun
    parse_type!("fun ()");
    parse_type!("fun ((...) -> number())");
    parse_type!("fun ((A, b) -> c:d())");

    // union
    parse_type!("10 | 1 + 2 | (foo | {a, b, c}) | baz");
}

#[test]
fn parse_form_works() {
    // module attribute
    parse_form!("-module(foo).");

    // export attribute
    parse_form!("-export([]).");
    parse_form!("-export([foo/0, bar/2]).");

    // export type attribute
    parse_form!("-export_type([foo/0]).");
    parse_form!("-export_type([foo/0, bar/2]).");

    // import attribute
    parse_form!("-import(foo, []).");
    parse_form!("-import(foo, [bar/0, baz/5]).");

    // file attribute>
    parse_form!(r#"-file("/path/to/file", 10)."#);

    // wild attribute
    parse_form!("-my_attr([1, {2, 3}, #{}]).");

    // spec
    parse_form!("-spec foo () -> ok.");
    parse_form!("-spec foo (a) -> ok; (b) -> err.");
    parse_form!("-spec foo (a) -> ok; (B) -> err when B :: integer().");
    parse_form!("-spec foo (A) -> ok when A :: integer(), is_subtype(A, list()).");

    // remote spec
    parse_form!("-spec foo:bar () -> ok.");
    parse_form!("-spec foo:bar (a) -> ok when B :: integer(); (B) -> err.");

    // callback
    parse_form!("-callback foo () -> ok.");
    parse_form!("-callback foo (a) -> ok; (b) -> err.");
    parse_form!("-callback foo (a) -> ok; (B) -> err when B :: integer().");

    // fun declaration
    parse_form!("foo () -> ok.");
    parse_form!("foo (A, {B, _}) -> A + B.");
    parse_form!("foo (A) when is_integer(A) -> ok; foo (B) -> {error, B}.");

    // record declaration
    parse_form!("-record(foo, {}).");
    parse_form!("-record(foo, {a, b, c}).");
    parse_form!("-record(foo, {a = 10, b :: integer(), c = d :: atom()}).");

    // type declaration
    parse_form!("-type foo() :: integer().");
    parse_form!("-type foo(A, B) :: {A, B}.");
    parse_form!("-opaque foo() :: integer().");
}
