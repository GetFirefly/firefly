-module(maps).

-export([run/0]).

run() -> maps:get(a, #{a => 1}) + maps:get(b, #{a => 1}, 5).
