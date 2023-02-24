%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: hello
-module(init).

-export([boot/1]).

boot(_) ->
    Fun = fun callee/1,
    call(Fun).

call(Fun) when is_function(Fun) ->
    Fun(hello).

callee(Arg) ->
    erlang:display(Arg).
