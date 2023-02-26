%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

-module(init).

-export([boot/1]).

boot(_) ->
  erlang:error(nope).
