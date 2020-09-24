-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test().

test() ->
  test(-127, 127).

test(Shift, Shift) ->
  test(Shift);
test(Shift, Final) ->
  test(Shift),
  test(Shift + 1, Final).

test(Shift) ->
  Integer = 1,
  LeftShifted = Integer bsr Shift,
  NegativeShift = -1 * Shift,
  RightShifted = Integer bsl NegativeShift,
  display(LeftShifted == RightShifted).

