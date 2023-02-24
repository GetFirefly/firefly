-module(lists).

-export([reverse/1, reverse/2]).
-nifs([reverse/2]).

%% Shadowed by erl_bif_types: lists:reverse/2
-spec reverse(List1, Tail) -> List2 when
      List1 :: [T],
      Tail :: term(),
      List2 :: [T],
      T :: term().

reverse(_, _) ->
    erlang:nif_error(undef).

%% reverse(L) reverse all elements in the list L. reverse/2 is now a BIF!

-spec reverse(List1) -> List2 when
      List1 :: [T],
      List2 :: [T],
      T :: term().

reverse([] = L) ->
    L;
reverse([_] = L) ->
    L;
reverse([A, B]) ->
    [B, A];
reverse([A, B | L]) ->
    lists:reverse(L, [B, A]).

