-module(init).
-export([start/0]).
-import(erlang, [apply/2, display/1]).
%-import(erlang, [display/1]).

start() ->
  from_fun().

from_fun() ->
  Fun = proxy(fun return_from_export/0),
  Result = Fun(),
  display(Result).

proxy(FunName) ->
    fun() -> FunName() end.

return_from_export() ->
  from_fun.
