-module(init).
-export([start/0]).
-import(erlang, [apply/2, display/1]).

start() ->
  from_fun().

from_fun() ->
  Fun = fun return_from_export/0,
  Return = apply(Fun, []),
  display(Return).


return_from_export() ->
  from_fun.
