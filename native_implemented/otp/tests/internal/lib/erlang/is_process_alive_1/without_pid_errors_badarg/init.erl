-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (Pid) when is_pid(Pid) -> ignore;
    (Term) -> test(Term)
  end).

test(Pid) ->
  test:caught(fun () ->
    is_process_alive(Pid)
  end).
