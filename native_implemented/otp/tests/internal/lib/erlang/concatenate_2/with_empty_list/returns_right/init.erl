-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun (Term) ->
    test(Term)
  end).

test(Right) ->
  Left = [],
  true = is_list(Left),
  0 = length(Left),
  Final = Left ++ Right,
  display(Final == Right).
