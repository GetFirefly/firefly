-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (List) when is_list(List) -> ignore;
    (Term) -> test(Term)
  end).

test(Right) ->
  LeftElement = left,
  Left = [LeftElement],
  display(Left ++ Right == [LeftElement | Right]).
