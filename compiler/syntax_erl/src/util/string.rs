#![allow(dead_code)]
use liblumen_binary::Endianness;
use liblumen_intern::Ident;

use super::encoding::Encoding;
use super::string_tokenizer::{StringTokenizeError, StringTokenizer};

pub fn string_to_codepoints(string: Ident) -> Result<Vec<u64>, StringTokenizeError> {
    StringTokenizer::new(string)
        .map(|v| v.map(|(cp, _span)| cp))
        .collect()
}

pub fn string_to_binary(
    ident: Ident,
    encoding: Encoding,
    endianness: Endianness,
) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();

    let tokenizer = StringTokenizer::new(ident);
    for tok in tokenizer {
        let (cp, span) = tok?;
        let encoded = encoding.encode(cp, span)?;
        encoded.write(endianness, &mut out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use liblumen_intern::Ident;

    use super::{string_to_binary, string_to_codepoints, Encoding, Endianness};

    #[test]
    fn string_literal_parse() {
        assert!(
            string_to_codepoints(Ident::from_str("abc")).unwrap()
                == vec!['a' as u64, 'b' as u64, 'c' as u64]
        );

        assert!(
            string_to_codepoints(Ident::from_str("a\\bc")).unwrap()
                == vec!['a' as u64, 8, 'c' as u64]
        );

        assert!(
            string_to_codepoints(Ident::from_str("a\\b\\d\\e\\f\\n\\r\\s\\t\\vc")).unwrap()
                == vec!['a' as u64, 8, 127, 27, 12, 10, 13, ' ' as u64, 9, 11, 'c' as u64]
        );

        assert!(
            string_to_codepoints(Ident::from_str("a\\'\\\"\\\\c")).unwrap()
                == vec!['a' as u64, '\'' as u64, '"' as u64, '\\' as u64, 'c' as u64]
        );

        assert!(
            string_to_codepoints(Ident::from_str("a\\1\\12\\123c")).unwrap()
                == vec!['a' as u64, 0o1, 0o12, 0o123, 'c' as u64]
        );
        assert!(string_to_codepoints(Ident::from_str("\\123")).unwrap() == vec![0o123]);
        assert!(string_to_codepoints(Ident::from_str("\\12")).unwrap() == vec![0o12]);
        assert!(string_to_codepoints(Ident::from_str("\\1")).unwrap() == vec![0o1]);

        assert!(
            string_to_codepoints(Ident::from_str("a\\xffc")).unwrap()
                == vec!['a' as u64, 0xff, 'c' as u64]
        );
        assert!(string_to_codepoints(Ident::from_str("\\xff")).unwrap() == vec![0xff]);

        assert!(string_to_codepoints(Ident::from_str("\\x{ff}")).unwrap() == vec![0xff]);
        assert!(string_to_codepoints(Ident::from_str("\\x{ffff}")).unwrap() == vec![0xffff]);

        assert!(string_to_codepoints(Ident::from_str("\\^a\\^z")).unwrap() == vec![1, 26]);
    }

    #[test]
    fn test_string_to_binary() {
        assert!(
            string_to_binary(Ident::from_str("abcå"), Encoding::Utf8, Endianness::Big).unwrap()
                == vec![0x61, 0x62, 0x63, 0xc3, 0xa5]
        )
    }
}
