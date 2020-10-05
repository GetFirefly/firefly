-module(init).
-export([start/0]).
-import(erlang, [load_nif/2]).

start() ->
  Path = "my_nif",
  LoadInfo = 0,
  test:caught(fun () ->
    load_nif(Path, LoadInfo)
  end).
