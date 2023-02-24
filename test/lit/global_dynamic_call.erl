%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile hello world

%% CHECK: {alice, says, hello, to, bob}
%% CHECK: {eve, overhears}

-module(init).

-export([boot/1, hello/2, listen/1]).

-import(erlang, [display/1]).

boot(_) ->
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
