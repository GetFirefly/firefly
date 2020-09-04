-module(test_small_integer).
-export([negative/0, zero/0, positive/0]).
-import(lumen, [is_small_integer/1]).

negative() ->
  Negative = -1,
  true = is_small_integer(Negative),
  true = (Negative < 0),
  Negative.

zero() ->
  Zero = 0,
  true = is_small_integer(Zero),
  Zero.

positive() ->
  Positive = 1,
  true = is_small_integer(Positive),
  true = (Positive > 0),
  Positive.
