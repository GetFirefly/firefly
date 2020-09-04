-module(test_float).
-export([negative/0, zero/0, positive/0]).
-import(lumen, [is_small_integer/1]).

negative() ->
  Negative = -1.2,
  true = is_float(Negative),
  true = (Negative < 0.0),
  Negative.

zero() ->
  Zero = 0.0,
  true = is_float(Zero),
  Zero.

positive() ->
  Positive = 3.4,
  true = is_float(Positive),
  true = (Positive > 0.0),
  Positive.
