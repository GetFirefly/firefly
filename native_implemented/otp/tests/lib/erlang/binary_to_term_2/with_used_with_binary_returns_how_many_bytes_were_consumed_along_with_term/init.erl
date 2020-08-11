-module(init).
-export([start/0]).
-import(erlang, [binary_to_term/2, display/1]).

start() ->
  Binary = <<131, 100, 0, 5, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100>>,

  %% Using only `used` portion of binary returns the same result
  Options = [used],
  Tuple = binary_to_term(Binary, Options),
  display(Tuple),
  Used = element(2, Tuple),
  SplitBinaryTuple = split_binary(Binary, Used),
  Prefix = element(1, SplitBinaryTuple),
  PrefixTuple = binary_to_term(Prefix, Options),
  display(PrefixTuple),
  display(Tuple == PrefixTuple),

  %% Without used returns only the term
  display(binary_to_term(Binary, [])).
