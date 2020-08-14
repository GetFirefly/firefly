// Only 0-10 to work around https://github.com/lumen/lumen/issues/512
test_stdout!(trailing_zeros_are_not_truncated, "\"12346\"\n\"12345.7\"\n\"12345.68\"\n\"12345.679\"\n\"12345.6789\"\n\"12345.67890\"\n\"12345.678900\"\n\"12345.6789000\"\n\"12345.67890000\"\n\"12345.678900000\"\n\"12345.6789000000\"\n");
