-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  %% Can't use `test:each` with guards as those would use `is_list`
  test([]),
  test([hd | tl]),
  test([1, 2]).

test(List) ->
  display(is_list(List)).
