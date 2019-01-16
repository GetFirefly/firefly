use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;

use crate::beam::reader::chunk;
use crate::beam::reader::chunk::Chunk;
use crate::beam::reader::chunk::StandardChunk;
use crate::beam::reader::parts;
use crate::beam::reader::BeamFile;
use crate::beam::reader::RawBeamFile;
use crate::beam::reader::Result;
use crate::beam::reader::StandardBeamFile;

#[test]
fn raw_chunks() {
    let beam = RawBeamFile::from_file(test_file("test.beam")).unwrap();

    // Chunk List
    let expected = vec![
        "Atom", "Code", "StrT", "ImpT", "ExpT", "FunT", "LitT", "LocT", "Attr", "CInf", "Abst",
        "Line",
    ];
    assert_eq!(expected, collect_id(&beam.chunks()));

    let expected = vec![
        "AtU8", "Code", "StrT", "ImpT", "ExpT", "FunT", "LitT", "LocT", "Attr", "CInf", "Dbgi",
        "Docs", "ExDp", // Elixir-specific, deprecated exports
        "Line",
    ];
    let beam = RawBeamFile::from_file(test_file("Elixir.Unicode.beam")).unwrap();
    assert_eq!(expected, collect_id(&beam.chunks()));
}

#[test]
fn standard_chunks() {
    use StandardChunk::*;
    macro_rules! find_chunk {
        ($beam:expr, $chunk:ident) => {
            $beam
                .chunks()
                .iter()
                .filter_map(|c| {
                    if let $chunk(ref x) = *c {
                        Some(x)
                    } else {
                        None
                    }
                })
                .nth(0)
                .unwrap()
        };
    }

    let beam = StandardBeamFile::from_file(test_file("test.beam")).unwrap();

    // Chunk List
    assert_eq!(
        vec![
            "Atom", "Code", "StrT", "ImpT", "ExpT", "FunT", "LitT", "LocT", "Attr", "CInf", "Abst",
            "Line",
        ],
        collect_id(&beam.chunks())
    );

    // Atom Chunk
    let atoms = &find_chunk!(beam, Atom).atoms;
    assert_eq!(
        vec![
            "test",
            "hello",
            "ok",
            "module_info",
            "erlang",
            "get_module_info",
            "-hello/1-fun-0-",
            "io",
            "format",
        ],
        atoms.iter().map(|a| &a.name).collect::<Vec<_>>()
    );

    // Code Chunk
    let code = find_chunk!(beam, Code);
    assert_eq!(16, code.info_size);
    assert_eq!(0, code.version);
    assert_eq!(153, code.opcode_max);
    assert_eq!(9, code.label_count);
    assert_eq!(4, code.function_count);
    assert_eq!(91, code.bytecode.len());

    // StrT Chunk
    let strt = find_chunk!(beam, StrT);
    assert_eq!(0, strt.strings.len());

    // ImpT Chunk
    let atom_name = |id| &atoms[id as usize - 1].name;
    let import_to_string = |i: &parts::Import| {
        format!(
            "{}:{}/{}",
            atom_name(i.module),
            atom_name(i.function),
            i.arity
        )
    };
    assert_eq!(
        vec![
            "erlang:get_module_info/1",
            "erlang:get_module_info/2",
            "io:format/2",
        ],
        find_chunk!(beam, ImpT)
            .imports
            .iter()
            .map(import_to_string)
            .collect::<Vec<_>>()
    );

    // ExpT Chunk
    let export_to_string =
        |e: &parts::Export| format!("{}/{}@{}", atom_name(e.function), e.arity, e.label);
    assert_eq!(
        vec!["module_info/1@6", "module_info/0@4", "hello/1@2"],
        find_chunk!(beam, ExpT)
            .exports
            .iter()
            .map(export_to_string)
            .collect::<Vec<_>>()
    );

    // FunT Chunk
    let fun_to_string = |f: &parts::Function| {
        format!(
            "{}/{}@{}.{}.{}.{}",
            atom_name(f.function),
            f.arity,
            f.label,
            f.index,
            f.num_free,
            f.old_uniq
        )
    };
    assert_eq!(
        vec!["-hello/1-fun-0-/1@8.0.1.38182595"],
        find_chunk!(beam, FunT)
            .functions
            .iter()
            .map(fun_to_string)
            .collect::<Vec<_>>()
    );

    // LitT Chunk
    assert_eq!(
        vec![13],
        find_chunk!(beam, LitT)
            .literals
            .iter()
            .map(|l| l.len())
            .collect::<Vec<_>>()
    );

    // LocT Chunk
    let local_to_string =
        |l: &parts::Local| format!("{}/{}@{}", atom_name(l.function), l.arity, l.label);
    assert_eq!(
        vec!["-hello/1-fun-0-/1@8"],
        find_chunk!(beam, LocT)
            .locals
            .iter()
            .map(local_to_string)
            .collect::<Vec<_>>()
    );

    // Attr Chunk
    assert_eq!(40, find_chunk!(beam, Attr).term.len());

    // CInf Chunk
    assert_eq!(209, find_chunk!(beam, CInf).term.len());

    // Abst Chunk
    assert_eq!(307, find_chunk!(beam, Abst).term.len());
}

enum EncodeTestChunk {
    Idempotent(chunk::StandardChunk),
    Other(chunk::RawChunk),
}
impl chunk::Chunk for EncodeTestChunk {
    fn id(&self) -> &chunk::Id {
        use self::EncodeTestChunk::*;
        match *self {
            Idempotent(ref c) => c.id(),
            Other(ref c) => c.id(),
        }
    }
    fn decode_data<R: Read>(id: &chunk::Id, reader: R) -> Result<Self>
    where
        Self: Sized,
    {
        use self::EncodeTestChunk::*;
        match id {
            b"LitT" => Ok(Other(chunk::RawChunk::decode_data(id, reader)?)),
            _ => Ok(Idempotent(chunk::StandardChunk::decode_data(id, reader)?)),
        }
    }
    fn encode_data<W: Write>(&self, writer: W) -> Result<()> {
        use self::EncodeTestChunk::*;
        match *self {
            Idempotent(ref c) => c.encode_data(writer),
            Other(ref c) => c.encode_data(writer),
        }
    }
}

#[test]
fn encode_chunks() {
    let mut original = Vec::new();
    std::io::copy(
        &mut File::open(test_file("test.beam")).unwrap(),
        &mut original,
    )
    .unwrap();

    let beam = BeamFile::<EncodeTestChunk>::from_reader(std::io::Cursor::new(&original)).unwrap();
    let mut encoded = Vec::new();
    beam.to_writer(&mut encoded).unwrap();

    assert_eq!(original, encoded);
}

fn test_file(name: &str) -> PathBuf {
    let mut path = PathBuf::from("tests/testdata/reader");
    path.push(name);
    path
}

fn collect_id<C: Chunk>(chunks: &[&C]) -> Vec<String> {
    chunks
        .iter()
        .map(|c| std::str::from_utf8(c.id()).unwrap().to_string())
        .collect()
}
