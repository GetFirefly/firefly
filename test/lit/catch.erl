%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: badarg
-module(init).

-export([boot/1]).

boot(_) ->
  {'EXIT', {Reason, _}} = (catch abs(non_number())),
  erlang:display(Reason).

non_number() ->
  atom.
