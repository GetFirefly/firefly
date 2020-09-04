-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun (Augend) ->
    without_number_augend(Augend)
  end).

without_number_augend(Augend) ->
  case Augend of
    AugendNumber when is_number(AugendNumber) ->
      ignore;
    AugendTerm ->
      test:caught(fun () ->
        AugendTerm + 0
      end)
  end.
