%% $ erlc +debug_info test.erl
-module(test).

-export([hello/1]).

-spec hello(term()) -> ok.
hello(Name) ->
    Hello = fun () -> io:format("Hello ~p!", [Name]) end,
    Hello(),
    ok.
