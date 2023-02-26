%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: "yep"
-module(init).

-export([boot/1]).

boot(Args) ->
  Bool = case Args of
             [] -> false;
             [_ | _] -> true
        end,
  case Bool of
      true ->
          erlang:display("yep");
      false ->
          erlang:display("nope")
  end.

