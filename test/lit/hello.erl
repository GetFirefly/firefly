%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile hello world
-module(init).

-export([boot/1]).

%% CHECK: <<"hello">>, <<"world">>
boot(Args) ->
    erlang:display(Args).
