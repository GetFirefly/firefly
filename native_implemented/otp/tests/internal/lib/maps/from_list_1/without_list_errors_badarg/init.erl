-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  test(tuple()),
  test(pid()),
  test(reference()),
  test(small_integer()),
  test(big_integer()),
  test(float()),
  test(atom()),
  test(binary()).

test(Term) ->
  try maps:from_list(Term) of
    Map -> display({map, Map})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.

tuple() ->
  {}.

pid() ->
  self().

reference() ->
  make_ref().

small_integer() ->
  SmallInteger = 0,
  true = is_small_integer(SmallInteger),
  SmallInteger.

big_integer() ->
  BigInteger = (1 bsl 46),
  true = is_big_integer(BigInteger),
  BigInteger.

float() ->
  0.0.

atom() ->
  atom.

binary() ->
  <<>>.
