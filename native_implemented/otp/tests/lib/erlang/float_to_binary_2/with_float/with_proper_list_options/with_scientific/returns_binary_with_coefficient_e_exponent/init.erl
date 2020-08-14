-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_binary/2]).

start() ->
  binaries().

binaries() ->
   binaries(0, 21).

binaries(MaxDigits, MaxDigits) ->
  binary(MaxDigits);
binaries(Digits, MaxDigits) ->
  binary(Digits),
  binaries(Digits + 1, MaxDigits).

binary(Digits) ->
  Scientific = {scientific, Digits},
  Options = [Scientific],
  display(float_to_binary(1234567890.0987654321, Options)).
