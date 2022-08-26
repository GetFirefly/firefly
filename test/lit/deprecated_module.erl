-module(deprecated_module).

-export([display/1]).
-deprecated({'_', '_', eventually}).

display(Arg) ->
    erlang:display(Arg).
