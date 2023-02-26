%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: ok
-module(init).

-export([boot/1]).

boot(_) ->
    F = fun A(X) ->
                if 
                    X > 10 ->
                       ok;
                    true ->
                        A(X + 1)
                end
        end,
    erlang:display(F(0)).
