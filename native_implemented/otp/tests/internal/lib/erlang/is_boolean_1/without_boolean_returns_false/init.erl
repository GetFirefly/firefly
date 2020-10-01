-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun (Term) ->
    display(is_boolean(Term))
  end).
