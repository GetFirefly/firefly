%% RUN: @firefly compile -C no_default_init --bin -o @tempfile @file && @tempfile

%% CHECK: ok
-module(init).

-export([boot/1]).

boot(_) -> a(10).

a(X) when X rem 10 == 0 ->
    erlang:display(<<"ok">>).
