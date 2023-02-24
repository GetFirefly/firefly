use super::*;
use crate::ops::*;

macro_rules! assert_opcode_match {
    ($lhs:expr, $rhs:expr) => {
        match (&$lhs, &$rhs) {
            (
                Opcode::LoadBinary(LoadBinary {
                    dest: dest1,
                    value: value1,
                }),
                Opcode::LoadBinary(LoadBinary {
                    dest: dest2,
                    value: value2,
                }),
            ) => {
                if core::ptr::eq(*value1, *value2) {
                    assert_eq!($lhs, $rhs)
                } else {
                    unsafe {
                        assert_eq!(&**value1, &**value2, "mismatched binaries");
                        assert_eq!(dest1, dest2);
                    }
                }
            }
            (
                Opcode::LoadBitstring(LoadBitstring {
                    dest: dest1,
                    value: value1,
                }),
                Opcode::LoadBitstring(LoadBitstring {
                    dest: dest2,
                    value: value2,
                }),
            ) => {
                if core::ptr::eq(*value1, *value2) {
                    assert_eq!($lhs, $rhs)
                } else {
                    unsafe {
                        assert_eq!(&**value1, &**value2, "mismatched bitstrings");
                        assert_eq!(dest1, dest2);
                    }
                }
            }
            (lhs, rhs) => assert_eq!(lhs, rhs),
        }
    };
}

#[test]
fn bytecode_atom_encoding_test() {
    let mut builder = Builder::new(StandardByteCode::new());
    builder.insert_atom("test");
    builder.insert_atom("main");
    let code = builder.finish();
    let mut buffer = Vec::new();
    let mut writer = BytecodeWriter::new(&mut buffer);
    writer.write_atoms(&code).unwrap();

    let mut reader = BytecodeReader::<AtomicStr, LocalAtomTable>::new(buffer.as_slice());
    reader.read_atoms(code.atoms.len()).unwrap();
    assert_eq!(reader.code.atoms.len(), 2);
    assert_eq!(reader.input.len(), 0);
    let atoms = reader.code.atoms.iter().collect::<Vec<_>>();
    let test = reader.code.atoms.get_or_insert("test").unwrap();
    let main = reader.code.atoms.get_or_insert("main").unwrap();
    assert!(atoms.contains(&test));
    assert!(atoms.contains(&main));
}

#[test]
fn bytecode_binary_encoding_test() {
    use firefly_binary::Encoding;

    let mut builder = Builder::new(StandardByteCode::new());
    builder.insert_binary("test".as_bytes(), Encoding::Utf8);
    builder.insert_binary("main".as_bytes(), Encoding::Utf8);
    let code = builder.finish();
    let mut buffer = Vec::new();
    let mut writer = BytecodeWriter::new(&mut buffer);
    writer.write_binaries(&code).unwrap();

    let mut reader = BytecodeReader::<AtomicStr, LocalAtomTable>::new(buffer.as_slice());
    reader.read_binaries(code.binaries.len()).unwrap();
    assert_eq!(reader.code.binaries.len(), 2);
    assert_eq!(reader.input.len(), 0);
    let bins = reader
        .code
        .binaries
        .iter()
        .map(|ptr| unsafe { str::from_utf8(ptr.as_ref().as_bytes()).unwrap() })
        .collect::<Vec<_>>();
    assert!(bins.contains(&"test"));
    assert!(bins.contains(&"main"));
}

#[test]
fn bytecode_function_encoding_test() {
    let mut builder = Builder::new(StandardByteCode::new());
    let test_main_1 = ModuleFunctionArity {
        module: builder.insert_atom("test"),
        function: builder.insert_atom("main"),
        arity: 1,
    };
    let function = builder.build_function(test_main_1, None).unwrap();
    function.finish();
    let code = builder.finish();

    let mut buffer = Vec::new();
    let mut writer = BytecodeWriter::new(&mut buffer);
    writer.write_atoms(&code).unwrap();
    writer.write_functions(&code).unwrap();

    let mut reader = BytecodeReader::<AtomicStr, LocalAtomTable>::new(buffer.as_slice());
    reader.read_atoms(code.atoms.len()).unwrap();
    reader.read_functions(code.functions.len()).unwrap();
    assert_eq!(reader.code.functions.len(), 1);
    assert_eq!(reader.input.len(), 0);
    assert_eq!(
        reader.code.function_by_mfa(&test_main_1),
        Some(&Function::Bytecode {
            id: 0,
            is_nif: false,
            mfa: test_main_1,
            offset: 3,
        })
    );
}

#[test]
fn bytecode_integration_test() {
    let code = generate_code();
    let mut buffer = Vec::new();
    let writer = BytecodeWriter::new(&mut buffer);
    writer.write(&code).unwrap();

    let reader = BytecodeReader::new(buffer.as_slice());
    let code2: ByteCode<AtomicStr, LocalAtomTable> = reader.read().unwrap();

    // Verify the contents of the parsed bytecode match the original
    assert_eq!(code.atoms.len(), code2.atoms.len());
    assert_eq!(code.atoms, code2.atoms);
    assert_eq!(code.binaries.len(), code2.binaries.len());
    assert_eq!(code.binaries, code2.binaries);
    assert_eq!(code.functions.len(), code2.functions.len());
    assert_eq!(code.functions, code2.functions);
    assert_eq!(code.code.len(), code2.code.len());

    for (op1, op2) in code.code.iter().zip(code2.code.iter()) {
        assert_opcode_match!(op1, op2);
    }
}

fn generate_code() -> ByteCode<AtomicStr, LocalAtomTable> {
    let mut builder = Builder::new(ByteCode::new());
    let test_main_1 = ModuleFunctionArity {
        module: builder.insert_atom("test"),
        function: builder.insert_atom("main"),
        arity: 1,
    };
    let mut function = builder.build_function(test_main_1, None).unwrap();
    let test_main_1 = function.get_or_define_function(test_main_1);

    let entry = function.create_block(1);
    let is_nil_block = function.create_block(0);
    let is_cons_block = function.create_block(2);

    // Build entry
    function.switch_to_block(entry);
    let arg = function.block_args(entry)[0];
    let is_nil = function.build_is_nil(arg);
    function.build_br_if(is_nil, is_nil_block, &[]);
    let (h, t) = function.build_split(arg);
    function.build_br(is_cons_block, &[h, t]);

    // Build fail case
    function.switch_to_block(is_nil_block);
    let badarg = function.build_atom("badarg");
    function.build_raise(ErrorKind::Error, badarg);

    // Build success case
    function.switch_to_block(is_cons_block);
    let (h, t) = {
        let args = function.block_args(is_cons_block);
        (args[0], args[1])
    };
    let tail = function.build_call(test_main_1, &[t]);
    let erlang = function.insert_atom("erlang");
    let mul = function.insert_atom("mul");
    let erlang_mul_2 = function.get_or_define_bif(ModuleFunctionArity {
        module: erlang,
        function: mul,
        arity: 2,
    });
    let two = function.build_int(2);
    let head = function.build_call(erlang_mul_2, &[h, two]);
    let cons = function.build_cons(head, tail);
    function.build_ret(cons);
    function.finish();
    builder.finish()
}
