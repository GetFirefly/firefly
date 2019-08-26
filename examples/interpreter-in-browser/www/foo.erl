-module(foo).

-export([bar/1]).

bar(Arg) ->
    lumen_intrinsics:println({yay, Arg}).
