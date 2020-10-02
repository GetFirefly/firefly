-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(test:atom()),
  test(test:binary()),
  test(test:function()),
  test(test:list()),
  test(test:map()),
  test(test:nil()),
  test(test:pid()),
  test(test:reference()),
  test(test:tuple()).

test(Term) ->
  display(is_number(Term)).
