%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile true

%% CHECK: found
-module(init).

-export([boot/1]).

boot(Args) ->
    Mapper = fun(X) -> contains(Args, X) end,
    Result = has_true([<<"true">>], Mapper),
    erlang:display(Result).

has_true([], Fn) when is_function(Fn) -> not_found;
has_true([Arg | Rest], Fn) when is_function(Fn) ->
    case Fn(Arg) of
        true -> found;
        _ -> has_true(Rest, Fn)
    end.

contains([], _) -> false;
contains([X | _], X) -> true;
contains([_ | Rest], X) -> contains(Rest, X).
