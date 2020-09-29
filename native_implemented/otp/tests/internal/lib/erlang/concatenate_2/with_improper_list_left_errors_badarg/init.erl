-module(init).
-export([start/0]).

start() ->
  Left = [hd | tl],
  true = is_list(Left),
  Right = [],
  true = is_list(Right),
  test:caught(fun () ->
    Left ++ Right
  end).
