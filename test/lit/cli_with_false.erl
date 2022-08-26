%% RUN: @firefly compile -o @tempfile @tests/cli.erl && @tempfile false

%% CHECK: <<"Nothing to say.">>
