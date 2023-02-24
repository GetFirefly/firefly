%% RUN: @firefly compile --bin -o @tempfile @tests/cli.erl @tests/lists.erl && @tempfile false

%% CHECK: <<"Nothing to say.">>
