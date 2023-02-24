-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Result = do_stuff(non_number),
  display(Result).

do_stuff(Reason) ->
  erlang:fail(Reason).
