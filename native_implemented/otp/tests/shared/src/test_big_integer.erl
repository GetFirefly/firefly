-module(test_big_integer).
-export([negative/0, positive/0]).
-import(lumen, [is_big_integer/1]).

negative() ->
  Negative = -1 * positive(),
  true = is_big_integer(Negative),
  true = (Negative < 0),
  Negative.

positive() ->
  Positive = (1 bsl 64),
  true = is_big_integer(Positive),
  true = (Positive > 0),
  Positive.
