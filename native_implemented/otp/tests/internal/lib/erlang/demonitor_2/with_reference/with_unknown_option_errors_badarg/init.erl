-module(init).
-export([start/0]).

start() ->
  Reference = test:reference(),
  %% Typo, correct option is `info`
  Options = [information],
  test:caught(fun () ->
    demonitor(Reference, Options)
  end).
