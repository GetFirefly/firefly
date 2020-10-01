-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(test:atom()),
  test(test:big_integer()),
  test(test:binary()),
  test(test:float()),
  test(test:function()),
  test(test:list()),
  test(test:nil()),
  test(test:pid()),
  test(test:reference()),
  test(test:small_integer()),
  test(test:tuple()).

test(Term) ->
  display(is_map(Term)).
