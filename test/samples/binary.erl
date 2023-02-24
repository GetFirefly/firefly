-module(init).

-export([start/0]).

-import(erlang, [print/1]).

start() ->
    N = 10,
    print(<<"Value is: ", N/integer>>),
    Acc = <<>>,
    print(Acc),
    write([1, 2, 3, 4, 5], Acc).

write([], Acc) ->
    print(Acc);
write([H|T], Acc) ->
    print(H),
    print(Acc),
    Char = $0 + H,
    write(T, <<Acc/binary, Char>>).
