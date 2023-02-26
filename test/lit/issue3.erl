%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: 3628800
-module(init).

-export([boot/1]).

boot(_) -> erlang:display(factorial(10)).

factorial(0) ->
    1;
factorial(N) when N > 0 ->
    N * factorial(N - 1).
