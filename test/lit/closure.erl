%% RUN: @lumen compile -o @tempfile @file && @tempfile

%% CHECK: hello
-module(init).

-export([boot/1]).

boot(Args) ->
    Fun = fun () -> callee(Args) end,
    call(Fun).

call(Fun) when is_function(Fun) ->
    Fun().

callee(Arg) ->
    erlang:display(Arg).
