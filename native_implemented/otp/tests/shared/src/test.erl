-module(test).
-export([atom/0, big_integer/0, binary/0, caught/1, each/1, float/0, function/0, list/0, map/0, nil/0, pid/0, reference/0, small_integer/0, tuple/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

caught(Fun) ->
  try apply(Fun, []) of
    Return ->
      display({returned, Return})
  catch
    Class:Exception ->
      display({caught, Class, Exception})
  end.

each(Fun) ->
  apply(Fun, [atom()]),
  apply(Fun, [big_integer()]),
  apply(Fun, [binary()]),
  apply(Fun, [float()]),
  apply(Fun, [function()]),
  apply(Fun, [list()]),
  apply(Fun, [map()]),
  apply(Fun, [nil()]),
  apply(Fun, [pid()]),
  apply(Fun, [reference()]),
  apply(Fun, [small_integer()]),
  apply(Fun, [tuple()]).

%% Types

atom() ->
  Atom = atom,
  true = is_atom(Atom),
  Atom.

big_integer() ->
  test_big_integer:positive().

binary() ->
  Binary = <<>>,
  true = is_binary(Binary),
  Binary.

float() ->
  test_float:zero().

function() ->
  Fun = fun () ->
    ok
  end,
  true = is_function(Fun),
  Fun.

list() ->
  List = [hd | tl],
  true = is_list(List),
  List.

map() ->
  Map = #{},
  true = is_map(Map),
  Map.

nil() ->
  Nil = [],
  true = is_list(Nil),
  0 = length(Nil),
  Nil.

pid() ->
  Pid = self(),
  true = is_pid(Pid),
  Pid.

reference() ->
  Reference = make_ref(),
  true = is_reference(Reference),
  Reference.

small_integer() ->
  test_small_integer:zero().

tuple() ->
  Tuple = {},
  true = is_tuple(Tuple),
  Tuple.
