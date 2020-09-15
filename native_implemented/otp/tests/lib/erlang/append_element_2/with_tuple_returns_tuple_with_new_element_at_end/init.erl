-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1]).

start() ->
  test:each(fun (Element) ->
    test(Element)
  end).

test(Element) ->
  AccTuple = {},
  FinalTuple = append_element(AccTuple, Element),
  display(tuple_size(FinalTuple) == tuple_size(AccTuple) + 1),
  display(element(tuple_size(AccTuple) + 1, FinalTuple) == Element).
