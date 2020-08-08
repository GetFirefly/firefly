// `without_atom_errors_badarg` in unit tests
// `with_atom_without_atom_encoding_errors_badarg` in unit tests
// `with_atom_with_atom_without_name_encoding_errors_badarg` in unit tests

test_stdout!(
    with_atom_with_encoding_atom_returns_name_in_binary,
    "<<\"one\">>\n<<\"two\">>\n<<\"three\">>\n"
);
