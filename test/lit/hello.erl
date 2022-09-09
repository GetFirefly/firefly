%% RUN: @firefly compile -C no_default_init --bin -o @tempfile @file && @tempfile hello world
-module(init).

-export([boot/1]).

%% CHECK: <<"hello">>, <<"world">>
boot(Args) ->
    erlang:display(Args).
