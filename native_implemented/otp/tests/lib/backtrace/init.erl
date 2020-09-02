-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  bad_reverse([0, 1, 2, 3]).

bad_reverse([H|T]) ->
  bad_reverse(T) ++ [H];
bad_reverse(L) ->
  [tl(L) | hd(L)].
