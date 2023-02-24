-module(init).
-export([start/0]).
-import(erlang, [atom_to_binary/2, make_ref/0, display/1, self/0]).
-import(lumen, [is_big_integer/1]).

start() ->
  with_non_empty_list(),
  with_tuple(),
  with_pid(),
  with_reference(),
  with_small_integer(),
  with_big_integer(),
  with_float(),
  with_binary(),
  with_closure(),
  with_empty_list().

with_non_empty_list() ->
  errors_badarg([non_empty]).

with_tuple() ->
  errors_badarg([]).

with_pid() ->
  errors_badarg(self()).

with_reference() ->
  errors_badarg(make_ref()).

with_small_integer() ->
  errors_badarg(1).

with_big_integer() ->
  BigInteger = 1 bsl 63,
  true = is_big_integer(BigInteger),
  errors_badarg(BigInteger).

with_float() ->
  errors_badarg(1.0).

with_binary() ->
  errors_badarg(<<>>).

with_closure() ->
  errors_badarg(fun () ->
    ok
  end).

with_empty_list() ->
  errors_badarg([]).

errors_badarg(NotAtom) ->
  display(is_atom(NotAtom)),
  try atom_to_binary(NotAtom, encoding()) of
    Binary ->
      display(Binary)
  catch
    error:Reason:Backtrace -> display(Reason)
  end.

encoding() ->
  unicode.
