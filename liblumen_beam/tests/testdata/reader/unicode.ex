defmodule Unicode do
  @moduledoc false

  def string(), do: "string"
  def ascii_atom(), do: :"atom"
  def utf8_atom(), do: :"åtom"

  def add1(n) when is_number(n) do
    n + 1
  end

  def add1(list) when is_list(list) do
    Enum.map(list, &add1/1)
  end
end
