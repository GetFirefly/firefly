-module(init).

-export([boot/1]).

boot(Args) ->
    Mapper = fun(X) -> contains(Args, X) end,
    has_true([<<"true">>], Mapper).

has_true([], Fn) when is_function(Fn) -> false;
has_true([Arg | Rest], Fn) when is_function(Fn) ->
    case Fn(Arg) of
        true -> true;
        _ -> has_true(Rest, Fn)
    end.

contains([], _) -> false;
contains([X | _], X) -> true;
contains([_ | Rest], X) -> contains(Rest, X).
