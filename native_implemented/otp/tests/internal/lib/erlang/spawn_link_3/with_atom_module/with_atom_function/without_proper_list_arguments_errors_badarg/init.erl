-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = erlang,
  Function = '+',
  Args = [1 | 2],
  try spawn_link(Module, Function, Args) of
    _ -> display({child, spawned})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.
