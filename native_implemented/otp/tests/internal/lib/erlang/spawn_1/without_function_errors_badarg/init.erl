-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  errors(atom()),
  errors(big_integer()),
  errors(binary()),
  errors(empty_list()),
  errors(float()),
  errors(map()),
  errors(non_empty_list()),
  errors(pid()),
  errors(reference()),
  errors(small_integer()),
  errors(tuple()).

errors(Fun) ->
  Try = try spawn(Fun) of
    Spawned -> {spawned, Spawned}
  catch
    Exception -> Exception
  end,
  display(Try).

atom() ->
  atom.

big_integer() ->
  BigInteger = (1 bsl 63),
  true = is_big_integer(BigInteger),
  BigInteger.

binary() ->
  <<"binary">>.

empty_list() ->
  [].

float() ->
  0.0.

%% work around MapPut not implemented
map() ->
  maps:from_list([]).

non_empty_list() ->
  [non_empty].

pid() ->
  self().

reference() ->
  make_ref().

small_integer() ->
  SmallInteger = 0,
  true = is_small_integer(SmallInteger),
  SmallInteger.

tuple() ->
  [].
