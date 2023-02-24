-module(init).

-export([start/0]).

-import(erlang, [print/1]).

start() ->
    N = 10,
    print(<<"Value is: ", N/integer>>).
