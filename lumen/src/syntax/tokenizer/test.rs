use crate::syntax::tokenizer::Tokenizer;

use trackable::*;

macro_rules! tokenize {
    ($text:expr) => {
        Tokenizer::new($text)
            .map(|t| track_try_unwrap!(t).text().to_string())
            .collect::<Vec<_>>();
    };
}

#[test]
fn tokenize_comments() {
    let src = "% foo";
    assert_eq!(tokenize!(src), ["% foo"]);

    let src = r#"
% foo
 % bar"#;
    assert_eq!(tokenize!(src), ["\n", "% foo", "\n", " ", "% bar"]);
}

#[test]
fn tokenize_numbers() {
    let src = "10 1.02";
    assert_eq!(tokenize!(src), ["10", " ", "1.02"]);
}

#[test]
fn tokenize_atoms() {
    let src = "foo 'BAR' comté äfunc";
    assert_eq!(
        tokenize!(src),
        ["foo", " ", "'BAR'", " ", "comté", " ", "äfunc"]
    );
}

#[test]
fn tokenize_variables() {
    let src = "Foo BAR _ _Baz";
    assert_eq!(tokenize!(src), ["Foo", " ", "BAR", " ", "_", " ", "_Baz"]);
}

#[test]
fn tokenize_strings() {
    let src = r#""foo" "b\tar""#;
    assert_eq!(tokenize!(src), [r#""foo""#, " ", r#""b\tar""#]);
}

#[test]
fn tokenize_chars() {
    let src = r#"$a $\t $\^a $\^]"#;
    assert_eq!(
        tokenize!(src),
        ["$a", " ", r#"$\t"#, " ", r#"$\^a"#, " ", r#"$\^]"#]
    );
}

#[test]
fn tokenize_module_declaration() {
    let src = "-module(foo).";
    assert_eq!(tokenize!(src), ["-", "module", "(", "foo", ")", "."]);
}
