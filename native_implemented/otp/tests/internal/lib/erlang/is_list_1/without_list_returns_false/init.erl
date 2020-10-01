-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  %% Can't use `test:each` with guards as those would use `is_list`
  test(test:atom()),
  test(test:big_integer()),
  test(test:binary()),
  test(test:float()),
  test(test:function()),
  test(test:map()),
  test(test:pid()),
  test(test:reference()),
  test(test:small_integer()),
  test(test:tuple()).

test(List) ->
  display(is_list(List)).
