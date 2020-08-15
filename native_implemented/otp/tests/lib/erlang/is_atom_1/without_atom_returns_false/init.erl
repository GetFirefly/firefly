-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  display(is_atom([1])),
  display(is_atom({})),
  display(is_atom(map())),
  display(is_atom(pid())),
  display(is_atom(reference())),
  display(is_atom(small_integer())),
  display(is_atom(big_integer())),
  display(is_atom(float())),
  display(is_atom(binary())).
%%  display(is_atom([])).

%% work around MapPut not implemented
map() ->
  maps:from_list([]).

pid() ->
  self().

reference() ->
  make_ref().

small_integer() ->
  SmallInteger = 0,
  true = is_small_integer(SmallInteger),
  SmallInteger.

big_integer() ->
  BigInteger = (1 bsl 63),
  true = is_big_integer(BigInteger),
  BigInteger.

float() ->
  0.0.

binary() ->
  <<"binary">>.
