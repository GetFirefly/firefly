-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(1, 2).

test(Dividend, Divisor) ->
  caught(fun () ->
    display({dividend, Dividend}),
    display({divisor, Divisor}),
    display(Dividend / Divisor)
  end).

caught(Fun) ->
    catch Fun().
