-module(init).
-export([start/0]).

start() ->
  test:caught(fun () ->
    element(1, {})
  end),
  test:caught(fun () ->
    element(-1, {1})
  end),
  test:caught(fun () ->
    element(0, {1})
  end),
  test:caught(fun () ->
    element(2, {1})
  end),
  test:caught(fun () ->
    element(-1, {1, 2})
  end),
  test:caught(fun () ->
    element(0, {1, 2})
  end),
  test:caught(fun () ->
    element(3, {1, 2})
  end).
