%% RUN: @firefly compile -C no_default_init --bin -o @tempfile @tests/cli.erl && @tempfile

%% CHECK: <<"Nothing to say.">>
