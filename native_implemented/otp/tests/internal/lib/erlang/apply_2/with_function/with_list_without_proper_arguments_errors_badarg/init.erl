-module(init).
-export([start/0]).

start() ->
  test:caught(fun () ->
    apply(
      fun (A, B) ->
        {A, B}
      end,
      [a | b]
    )
  end).
