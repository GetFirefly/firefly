%% RUN: @firefly compile -C no_default_init --bin -o @tempfile @file && @tempfile

%% CHECK: 4
-module(init).

-export([boot/1]).

boot(_) -> a(10, 2, 3).

a(X, Y, Z) ->
    erlang:display((X + Y) div Z).
