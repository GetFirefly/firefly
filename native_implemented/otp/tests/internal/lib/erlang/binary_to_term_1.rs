// `without_binary_errors_badarg` in unit tests

test_stdout!(with_binary_encoding_atom_returns_atom, "atom\n");
test_stdout!(with_binary_encoding_empty_list_returns_empty_list, "[]\n");
test_stdout!(with_binary_encoding_list_returns_list, "[zero, 1]\n");
test_stdout!(
    with_binary_encoding_small_integer_returns_small_integer,
    "0\n"
);
test_stdout!(
    with_binary_encoding_integer_returns_integer,
    "-2147483648\n"
);
test_stdout!(with_binary_encoding_new_float_returns_float, "1.0\n");
test_stdout!(
    with_binary_encoding_small_tuple_returns_tuple,
    "{zero, 1}\n"
);
test_stdout!(with_binary_encoding_byte_list_returns_list, "\"01\"\n");
test_stdout!(with_binary_encoding_binary_returns_binary, "<<0,1>>\n");
test_stdout!(
    with_binary_encoding_small_big_integer_returns_big_integer,
    "4294967295\n"
);
test_stdout!(
    with_binary_encoding_bit_string_returns_subbinary,
    "<<1,2:3>>\n"
);
test_stdout!(with_binary_encoding_small_atom_utf8_returns_atom, "'😈'\n");
