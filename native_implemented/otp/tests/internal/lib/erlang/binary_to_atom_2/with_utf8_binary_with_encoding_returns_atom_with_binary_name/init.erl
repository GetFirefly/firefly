-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
%%  display(binary_to_atom(<<"latin1">>, latin1)),
%%  display(<<"José">>),
  display(<<"José"/utf8>>),
  display(binary_to_atom(<<"José"/utf8>>, unicode)),
  display(binary_to_atom(<<"José"/utf8>>, utf8)).
