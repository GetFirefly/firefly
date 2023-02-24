%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: {ok, captured}
-module(init).

-export([boot/1]).

boot(_Args) ->
    Msg = {ok, captured},
    Fun = fun () -> callee(Msg) end,
    call(Fun).

call(Fun) when is_function(Fun) ->
    Fun().

callee(Arg) ->
    erlang:display(Arg).
