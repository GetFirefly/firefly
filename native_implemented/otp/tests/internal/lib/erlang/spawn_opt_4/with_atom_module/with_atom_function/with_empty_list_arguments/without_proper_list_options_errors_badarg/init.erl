-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = erlang,
  Function = '+',
  Args = [],
  Options = [link | monitor],
  try spawn_opt(Module, Function, Args, Options) of
    {ChildPid, ChildSpawnMonitor} ->
      display({child, ChildPid, spawned, with, monitor, ChildSpawnMonitor})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.


