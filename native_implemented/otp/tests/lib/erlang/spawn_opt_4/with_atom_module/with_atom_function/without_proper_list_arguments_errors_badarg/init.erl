-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = erlang,
  Function = '+',
  Args = [0 | 1],
  Options = [],
  try spawn_opt(Module, Function, Args, Options) of
    ChildPid ->
      display({child, ChildPid, spawned})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.


