-module(run).

-export([run/0]).

run() ->
    Ast = 'Elixir.TemplateDemo.Ast':ast(),
    'Elixir.DbMonDemo.AppSupervisor':start_link(Ast).
