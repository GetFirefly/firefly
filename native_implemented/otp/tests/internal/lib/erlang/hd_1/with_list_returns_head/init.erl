-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun (H) ->
    test(H)
  end).

test(H) ->
  test:each(fun (T) ->
    test(H, T)
  end).

test(ExpectedH, T) ->
  List = [ExpectedH | T],
  ActualH = hd(List),
  display(ActualH == ExpectedH).
