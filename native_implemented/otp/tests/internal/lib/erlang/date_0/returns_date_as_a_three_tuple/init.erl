-module(init).
-export([start/0]).
-import(erlang, [date/0, display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Date = date(),
  display(tuple_size(Date) == 3),
  {Year, Month, Day} = Date,
  display(2020 =< Year),
  display((1 =< Month) and (Month =< 12)),
  display((1 =< Day) and (Day =< 31)).

