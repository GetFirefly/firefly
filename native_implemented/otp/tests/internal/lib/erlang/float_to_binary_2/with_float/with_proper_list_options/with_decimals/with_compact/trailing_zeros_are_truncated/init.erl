-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_binary/2]).

start() ->
  binaries().

binaries() ->
   binaries(0, 16).

binaries(MaxDigits, MaxDigits) ->
  binary(MaxDigits);
binaries(Digits, MaxDigits) ->
  binary(Digits),
  binaries(Digits + 1, MaxDigits).

binary(Digits) ->
  Decimals = {decimals, Digits},
  Options = [Decimals, compact],
  display(float_to_binary(12345.6789, Options)).
