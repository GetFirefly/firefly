-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  {'EXIT', {Reason, _}} = (catch abs(non_number())),
  display(Reason).

%% TODO generate random non-number term
non_number() ->
  atom.
