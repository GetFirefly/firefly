-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = <<"erlang">>,
  Function = '+',
  Args = [0, 1],
  try spawn_monitor(Module, Function, Args) of
    {ChildPid, ChildSpawnMonitor} ->
      display({child, ChildPid, spawned, with, monitor, ChildSpawnMonitor})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.


