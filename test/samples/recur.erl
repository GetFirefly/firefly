-module(init).

-export([start/0]).

-import(erlang, [print/1]).

-spec start() -> ok | error.
start() ->
    loop_start().

loop_start() ->
    loop_a(0, 1000),
    print(<<"Done!">>).

loop_a(N, Max) when N < Max ->
    print(N),
    loop_b(N + 1, Max, Max - 1);
loop_a(_, _) ->
    ok.

loop_b(N, Max, _Min) when N < Max ->
    print(N),
    loop_a(N + 1, Max);
loop_b(_, _, _) ->
    ok.
