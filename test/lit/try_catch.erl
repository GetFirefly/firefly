%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: {error, badarg, [{init, boot, [<<"wrong">>]
-module(init).

-export([boot/1]).

boot(_) ->
    try spawn(<<"wrong">>) of
        List ->
            erlang:display(List),
            false
    catch
        Kind:Reason:Trace ->
            erlang:display({Kind, Reason, Trace}),
            true
    end.
