-module(fib).

-export([run/0]).

run() ->
    N = fib(8).

fib(X) when X < 2 -> 1;
fib(X) -> fib(X - 1) + fib(X-2).
