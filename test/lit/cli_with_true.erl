%% RUN: @firefly compile --bin -o @tempfile @tests/cli.erl @tests/lists.erl && @tempfile true

%% CHECK: <<"Hello, world!">>
