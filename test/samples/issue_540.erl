-module(call).
-export([woo/1]).

woo(A) -> A().
