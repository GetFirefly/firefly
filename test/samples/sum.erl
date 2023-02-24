-module(init).

-export([start/0]).

-import(erlang, [print/1]).

start() ->
    sum(10).

sum(N) ->
    sum(N, 0).

sum(0, Acc) ->
    print(<<"Result: ", Acc/integer>>);
sum(N, Acc) ->
    print(<<"Spawning adder for: ", N/integer>>),
    Parent = self(),
    spawn(fun() -> add(1, Acc, Parent) end),
    receive
        {result, Res} ->
            sum(N - 1, Res)
    end.

add(X, Y, From) ->
    From ! {result, X + Y}.
