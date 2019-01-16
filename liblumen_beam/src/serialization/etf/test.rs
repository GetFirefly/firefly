use std::io::Cursor;

use crate::serialization::etf::convert::TryInto;
use crate::serialization::etf::*;

#[test]
fn atom_test() {
    // Display
    assert_eq!("'foo'", Atom::from("foo").to_string());
    assert_eq!(r#"'fo\'o'"#, Atom::from(r#"fo'o"#).to_string());
    assert_eq!(r#"'fo\\o'"#, Atom::from(r#"fo\o"#).to_string());

    // Decode
    assert_eq!(
        Ok(Atom::from("foo")),
        decode(&[131, 100, 0, 3, 102, 111, 111]).try_into()
    ); // ATOM_EXT
    assert_eq!(
        Ok(Atom::from("foo")),
        decode(&[131, 115, 3, 102, 111, 111]).try_into()
    ); // SMALL_ATOM_EXT
    assert_eq!(
        Ok(Atom::from("foo")),
        decode(&[131, 118, 0, 3, 102, 111, 111]).try_into()
    ); // ATOM_UTF8_EXT
    assert_eq!(
        Ok(Atom::from("foo")),
        decode(&[131, 119, 3, 102, 111, 111]).try_into()
    ); // SMALL_ATOM_UTF8_EXT

    // Encode
    assert_eq!(
        vec![131, 100, 0, 3, 102, 111, 111],
        encode(Term::from(Atom::from("foo")))
    );
}

#[test]
fn integer_test() {
    // Display
    assert_eq!("123", FixInteger::from(123).to_string());
    assert_eq!("123", BigInteger::from(123).to_string());
    assert_eq!("-123", FixInteger::from(-123).to_string());
    assert_eq!("-123", BigInteger::from(-123).to_string());

    // Decode
    assert_eq!(Ok(FixInteger::from(10)), decode(&[131, 97, 10]).try_into()); // SMALL_INTEGER_EXT
    assert_eq!(
        Ok(FixInteger::from(1000)),
        decode(&[131, 98, 0, 0, 3, 232]).try_into()
    ); // INTEGER_EXT
    assert_eq!(
        Ok(FixInteger::from(-1000)),
        decode(&[131, 98, 255, 255, 252, 24]).try_into()
    ); // INTEGER_EXT
    assert_eq!(
        Ok(BigInteger::from(0)),
        decode(&[131, 110, 1, 0, 0]).try_into()
    ); // SMALL_BIG_EXT
    assert_eq!(
        Ok(BigInteger::from(513)),
        decode(&[131, 110, 2, 0, 1, 2]).try_into()
    ); // SMALL_BIG_EXT
    assert_eq!(
        Ok(BigInteger::from(-513)),
        decode(&[131, 110, 2, 1, 1, 2]).try_into()
    ); // SMALL_BIG_EXT
    assert_eq!(
        Ok(BigInteger::from(513)),
        decode(&[131, 111, 0, 0, 0, 2, 0, 1, 2]).try_into()
    ); // LARGE_BIG_EXT

    // Encode
    assert_eq!(vec![131, 97, 0], encode(Term::from(FixInteger::from(0))));
    assert_eq!(
        vec![131, 98, 255, 255, 255, 255],
        encode(Term::from(FixInteger::from(-1)))
    );
    assert_eq!(
        vec![131, 98, 0, 0, 3, 232],
        encode(Term::from(FixInteger::from(1000)))
    );
    assert_eq!(
        vec![131, 110, 1, 0, 0],
        encode(Term::from(BigInteger::from(0)))
    );
    assert_eq!(
        vec![131, 110, 1, 1, 10],
        encode(Term::from(BigInteger::from(-10)))
    );
    assert_eq!(
        vec![131, 110, 5, 0, 0, 228, 11, 84, 2],
        encode(Term::from(BigInteger::from(10000000000u64)))
    );
}

#[test]
fn float_test() {
    // Display
    assert_eq!("123", Float::from(123.0).to_string());
    assert_eq!("123.4", Float::from(123.4).to_string());
    assert_eq!("-123.4", Float::from(-123.4).to_string());

    // Decode
    assert_eq!(
        Ok(Float::from("1.23".parse::<f32>().unwrap() as f64)),
        decode(&[
            131, 99, 49, 46, 50, 50, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 56,
            50, 50, 52, 101, 43, 48, 48, 0, 0, 0, 0, 0
        ])
        .try_into()
    ); // FLOAT_EXT

    assert_eq!(
        Ok(Float::from(123.456)),
        // NEW_FLOAT_EXT
        decode(&[131, 70, 64, 94, 221, 47, 26, 159, 190, 119]).try_into()
    );
    assert_eq!(
        Ok(Float::from(-123.456)),
        // NEW_FLOAT_EXT
        decode(&[131, 70, 192, 94, 221, 47, 26, 159, 190, 119]).try_into()
    );
    // Encode
    assert_eq!(
        vec![131, 70, 64, 94, 221, 47, 26, 159, 190, 119],
        encode(Term::from(Float::from(123.456)))
    );
}

#[test]
fn pid_test() {
    // Display
    assert_eq!(
        r#"<'nonode@nohost'.1.2>"#,
        Pid::from(("nonode@nohost", 1, 2)).to_string()
    );

    // Decode
    assert_eq!(
        Ok(Pid::from(("nonode@nohost", 49, 0))),
        decode(&[
            131, 103, 100, 0, 13, 110, 111, 110, 111, 100, 101, 64, 110, 111, 104, 111, 115, 116,
            0, 0, 0, 49, 0, 0, 0, 0, 0
        ])
        .try_into()
    ); // PID_EXT

    // Encode
    assert_eq!(
        vec![
            131, 103, 100, 0, 13, 110, 111, 110, 111, 100, 101, 64, 110, 111, 104, 111, 115, 116,
            0, 0, 0, 49, 0, 0, 0, 0, 0
        ],
        encode(Term::from(Pid::from(("nonode@nohost", 49, 0))))
    );
}

#[test]
fn port_test() {
    // Display
    assert_eq!(
        r#"#Port<'nonode@nohost'.1>"#,
        Port::from(("nonode@nohost", 1)).to_string()
    );

    // Decode
    assert_eq!(
        Ok(Port::from(("nonode@nohost", 366))),
        decode(&[
            131, 102, 100, 0, 13, 110, 111, 110, 111, 100, 101, 64, 110, 111, 104, 111, 115, 116,
            0, 0, 1, 110, 0
        ])
        .try_into()
    ); // PORT_EXT

    // Encode
    assert_eq!(
        vec![
            131, 102, 100, 0, 13, 110, 111, 110, 111, 100, 101, 64, 110, 111, 104, 111, 115, 116,
            0, 0, 1, 110, 0
        ],
        encode(Term::from(Port::from(("nonode@nohost", 366))))
    );
}

#[test]
fn reference_test() {
    // Display
    assert_eq!(
        r#"#Ref<'nonode@nohost'.1>"#,
        Reference::from(("nonode@nohost", 1)).to_string()
    );

    // Decode
    assert_eq!(
        Ok(Reference::from(("nonode@nohost", vec![138016, 262145, 0]))),
        decode(&[
            131, 114, 0, 3, 100, 0, 13, 110, 111, 110, 111, 100, 101, 64, 110, 111, 104, 111, 115,
            116, 0, 0, 2, 27, 32, 0, 4, 0, 1, 0, 0, 0, 0
        ])
        .try_into()
    ); // NEW_REFERENCE_EXT
    assert_eq!(
        Ok(Reference::from(("foo", vec![2]))),
        // NEW_REFERENCE_EXT
        decode(&[131, 101, 115, 3, 102, 111, 111, 0, 0, 0, 2, 0]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 114, 0, 1, 100, 0, 3, 102, 111, 111, 0, 0, 0, 0, 123],
        encode(Term::from(Reference::from(("foo", 123))))
    );
}

#[test]
fn external_fun_test() {
    // Display
    assert_eq!(
        r#"fun 'foo':'bar'/3"#,
        ExternalFun::from(("foo", "bar", 3)).to_string()
    );

    // Decode
    assert_eq!(
        Ok(ExternalFun::from(("foo", "bar", 3))),
        decode(&[131, 113, 100, 0, 3, 102, 111, 111, 100, 0, 3, 98, 97, 114, 97, 3]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 113, 100, 0, 3, 102, 111, 111, 100, 0, 3, 98, 97, 114, 97, 3],
        encode(Term::from(ExternalFun::from(("foo", "bar", 3))))
    );
}

#[test]
fn internal_fun_test() {
    let term = InternalFun::New {
        module: Atom::from("a"),
        arity: 1,
        pid: Pid::from(("nonode@nohost", 36, 0)),
        index: 0,
        uniq: [
            115, 60, 203, 97, 151, 228, 98, 75, 71, 169, 49, 166, 34, 126, 65, 11,
        ],
        old_index: 0,
        old_uniq: 60417627,
        free_vars: vec![Term::from(FixInteger::from(10))],
    };
    let bytes = [
        131, 112, 0, 0, 0, 68, 1, 115, 60, 203, 97, 151, 228, 98, 75, 71, 169, 49, 166, 34, 126,
        65, 11, 0, 0, 0, 0, 0, 0, 0, 1, 100, 0, 1, 97, 97, 0, 98, 3, 153, 230, 91, 103, 100, 0, 13,
        110, 111, 110, 111, 100, 101, 64, 110, 111, 104, 111, 115, 116, 0, 0, 0, 36, 0, 0, 0, 0, 0,
        97, 10,
    ];
    // Decode
    assert_eq!(Ok(term.clone()), decode(&bytes).try_into());

    // Encode
    assert_eq!(Vec::from(&bytes[..]), encode(Term::from(term)));
}

#[test]
fn binary_test() {
    // Display
    assert_eq!("<<1,2,3>>", Binary::from(vec![1, 2, 3]).to_string());

    // Decode
    assert_eq!(
        Ok(Binary::from(vec![1, 2, 3])),
        decode(&[131, 109, 0, 0, 0, 3, 1, 2, 3]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 109, 0, 0, 0, 3, 1, 2, 3],
        encode(Term::from(Binary::from(vec![1, 2, 3])))
    );
}

#[test]
fn bit_binary_test() {
    // Display
    assert_eq!("<<1,2,3>>", BitBinary::from((vec![1, 2, 3], 8)).to_string());
    assert_eq!("<<1,2>>", BitBinary::from((vec![1, 2, 3], 0)).to_string());
    assert_eq!(
        "<<1,2,3:5>>",
        BitBinary::from((vec![1, 2, 3], 5)).to_string()
    );

    // Decode
    assert_eq!(
        Ok(BitBinary::from((vec![1, 2, 3], 5))),
        decode(&[131, 77, 0, 0, 0, 3, 5, 1, 2, 24]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 77, 0, 0, 0, 3, 5, 1, 2, 24],
        encode(Term::from(BitBinary::from((vec![1, 2, 3], 5))))
    );
}

#[test]
fn list_test() {
    // Display
    assert_eq!(
        "['a',1]",
        List::from(vec![
            Term::from(Atom::from("a")),
            Term::from(FixInteger::from(1))
        ])
        .to_string()
    );
    assert_eq!("[]", List::nil().to_string());

    // Decode
    assert_eq!(Ok(List::nil()), decode(&[131, 106]).try_into()); // NIL_EXT
    assert_eq!(
        Ok(List::from(vec![
            Term::from(FixInteger::from(1)),
            Term::from(FixInteger::from(2))
        ])),
        decode(&[131, 107, 0, 2, 1, 2]).try_into()
    ); // STRING_EXT
    assert_eq!(
        Ok(List::from(vec![Term::from(Atom::from("a"))])),
        decode(&[131, 108, 0, 0, 0, 1, 100, 0, 1, 97, 106]).try_into()
    );

    // Encode
    assert_eq!(vec![131, 106], encode(Term::from(List::nil())));
    assert_eq!(
        vec![131, 107, 0, 2, 1, 2],
        encode(Term::from(List::from(vec![
            Term::from(FixInteger::from(1)),
            Term::from(FixInteger::from(2))
        ])))
    );
    assert_eq!(
        vec![131, 108, 0, 0, 0, 1, 100, 0, 1, 97, 106],
        encode(Term::from(List::from(vec![Term::from(Atom::from("a"))])))
    );
}

#[test]
fn improper_list_test() {
    // Display
    assert_eq!(
        "[0,'a'|1]",
        ImproperList::from((
            vec![Term::from(FixInteger::from(0)), Term::from(Atom::from("a"))],
            Term::from(FixInteger::from(1))
        ))
        .to_string()
    );

    // Decode
    assert_eq!(
        Ok(ImproperList::from((
            vec![Term::from(Atom::from("a"))],
            Term::from(FixInteger::from(1))
        ))),
        decode(&[131, 108, 0, 0, 0, 1, 100, 0, 1, 97, 97, 1]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 108, 0, 0, 0, 1, 100, 0, 1, 97, 97, 1],
        encode(Term::from(ImproperList::from((
            vec![Term::from(Atom::from("a"))],
            Term::from(FixInteger::from(1))
        ))))
    );
}

#[test]
fn tuple_test() {
    // Display
    assert_eq!(
        "{'a',1}",
        Tuple::from(vec![
            Term::from(Atom::from("a")),
            Term::from(FixInteger::from(1))
        ])
        .to_string()
    );
    assert_eq!("{}", Tuple::from(vec![]).to_string());

    // Decode
    assert_eq!(
        Ok(Tuple::from(vec![
            Term::from(Atom::from("a")),
            Term::from(FixInteger::from(1))
        ])),
        decode(&[131, 104, 2, 100, 0, 1, 97, 97, 1]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 104, 2, 100, 0, 1, 97, 97, 1],
        encode(Term::from(Tuple::from(vec![
            Term::from(Atom::from("a")),
            Term::from(FixInteger::from(1))
        ])))
    );
}

#[test]
fn map_test() {
    let map = Map::from(vec![
        (
            Term::from(FixInteger::from(1)),
            Term::from(FixInteger::from(2)),
        ),
        (Term::from(Atom::from("a")), Term::from(Atom::from("b"))),
    ]);

    // Display
    assert_eq!("#{1=>2,'a'=>'b'}", map.to_string());

    assert_eq!("#{}", Map::from(vec![]).to_string());

    // Decode
    assert_eq!(
        Ok(map.clone()),
        decode(&[131, 116, 0, 0, 0, 2, 97, 1, 97, 2, 100, 0, 1, 97, 100, 0, 1, 98]).try_into()
    );

    // Encode
    assert_eq!(
        vec![131, 116, 0, 0, 0, 2, 97, 1, 97, 2, 100, 0, 1, 97, 100, 0, 1, 98],
        encode(Term::from(map))
    );
}

#[test]
fn compressed_term_test() {
    // Decode
    assert_eq!(
        Ok(List::from(
            (1..257)
                .map(|i| Term::from(FixInteger::from(i)))
                .collect::<Vec<_>>()
        )),
        decode(&[
            131, 80, 0, 0, 2, 9, 120, 218, 21, 210, 3, 187, 16, 6, 0, 0, 192, 151, 237, 150, 173,
            101, 219, 54, 182, 236, 186, 220, 235, 101, 219, 182, 237, 150, 93, 219, 178, 109, 219,
            182, 237, 175, 251, 13, 23, 20, 16, 16, 44, 64, 48, 193, 133, 16, 82, 40, 161, 133, 17,
            86, 56, 225, 69, 16, 81, 36, 145, 69, 17, 85, 52, 209, 197, 16, 211, 31, 98, 137, 45,
            142, 184, 226, 137, 47, 129, 132, 18, 73, 44, 137, 164, 146, 73, 46, 133, 148, 82, 249,
            83, 106, 105, 164, 149, 78, 122, 25, 100, 148, 73, 102, 89, 100, 149, 77, 118, 57, 228,
            148, 75, 110, 121, 228, 149, 79, 126, 5, 20, 84, 72, 97, 69, 20, 85, 76, 113, 37, 148,
            84, 74, 105, 101, 148, 85, 78, 121, 21, 84, 84, 201, 95, 254, 86, 89, 21, 85, 85, 83,
            93, 13, 53, 213, 82, 91, 29, 117, 213, 83, 95, 3, 13, 209, 72, 99, 77, 52, 213, 76,
            115, 45, 180, 20, 168, 149, 32, 173, 181, 209, 86, 59, 237, 117, 208, 81, 39, 157, 117,
            209, 85, 55, 221, 245, 208, 83, 47, 189, 245, 209, 87, 63, 253, 13, 48, 208, 32, 131,
            13, 49, 212, 48, 195, 141, 48, 210, 40, 163, 141, 49, 214, 56, 227, 77, 48, 209, 36,
            147, 77, 49, 213, 52, 211, 205, 48, 211, 44, 179, 205, 49, 215, 60, 243, 45, 176, 208,
            34, 255, 88, 108, 137, 165, 150, 89, 110, 133, 149, 86, 89, 109, 141, 181, 214, 89,
            111, 131, 141, 254, 245, 159, 255, 109, 178, 217, 22, 91, 109, 179, 221, 14, 59, 237,
            178, 219, 30, 123, 237, 179, 223, 1, 7, 29, 114, 216, 17, 71, 29, 115, 220, 9, 39, 157,
            114, 218, 25, 103, 157, 115, 222, 5, 23, 93, 114, 217, 21, 87, 93, 115, 221, 13, 55,
            221, 114, 219, 29, 119, 221, 115, 223, 3, 15, 61, 242, 216, 19, 79, 61, 243, 220, 11,
            47, 189, 242, 218, 27, 111, 189, 243, 222, 7, 31, 125, 242, 217, 23, 95, 125, 243, 221,
            15, 63, 27, 253, 46, 16, 248, 11, 162, 195, 225, 90
        ])
        .try_into()
    );
}

fn encode(term: Term) -> Vec<u8> {
    let mut buf = Vec::new();
    term.encode(&mut buf).unwrap();
    buf
}

fn decode(bytes: &[u8]) -> Term {
    Term::decode(Cursor::new(bytes)).unwrap()
}
