test_stdout!(
    without_list_right_returns_improper_list_with_right_as_tail,
    "true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\n"
);
test_stdout!(
    with_improper_list_right_returns_improper_list_with_right_as_tail,
    "[left, right_hd | right_tail]\n"
);
test_stdout!(
    with_list_right_returns_proper_list_with_right_as_tail,
    "[1, 2]\n[1, 2, 3]\n[1, 2, 3, 4]\n"
);
