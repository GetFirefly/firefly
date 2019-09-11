-module(fib).

-export([run/0]).

run() ->
    N = fib(8),
    lumen_intrinsics:println(N),
    N.

fib(0) -> 0;
fib(1) -> 1;
fib(X) -> fib(X-1) + fib(X-2).
