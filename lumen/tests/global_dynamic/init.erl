-module(init).
-export([hello/2, listen/1, start/0]).
-import(erlang, [display/1, print/1]).

start() ->
  global_dynamic(init, hello, [alice, bob]),
  global_dynamic(init, listen, [eve]).


global_dynamic(M, F, [A]) ->
  M:F(A);
global_dynamic(M, F, [A, B]) ->
  M:F(A, B).

hello(Speaker, Listener) ->
  display({Speaker, says, hello, to, Listener}).

listen(Listener) ->
  display({Listener, overhears}).
