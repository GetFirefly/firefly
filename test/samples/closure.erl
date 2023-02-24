-module(init).

-export([start/0]).


-import(erlang, [print/1]).

start() ->
    Name = void_map(<<"Paul">>),
    void(),
    Callback = 
        fun(Greeting) ->
            print(Greeting),
            print(Name)
        end,
    greet(Callback).

greet(Greeter) ->
    Greeter(<<"Hello">>).

void() ->
    ok.

void_map(A) ->
    A.
