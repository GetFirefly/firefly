defmodule Lumen.OTP.Log.Parser do
  @moduledoc """
  Documentation for `Lumen.OTP.Log.Parser`.
  """

  def parse(path) do
    path
    |> File.stream!()
      # drop until the first `failures:\n` that is the start of the failure details
    |> Stream.drop_while(fn
      "failures:\n" -> false
      _ -> true
    end)
    |> Stream.drop(2)
      # take while the second `failures:\n` that is summary of test names is shown and details end
    |> Stream.take_while(fn
      "failures:\n" -> false
      _ -> true
    end)
    |> Stream.filter(fn
      # remove common status code
      "Status code: 101\n" -> false
      # remove common signal (none)
      "Signal: ," <> _ -> false
      _ -> true
    end)
    |> Stream.chunk_while(
         {nil, []},
         fn
           "---- lumen::otp::" <> suffix, {previous_test, previous_test_lines} ->
             test =
               suffix
               |> String.split(" ", parts: 2)
               |> hd()

             {:cont, {previous_test, Enum.reverse(previous_test_lines)}, {test, []}}

           line, {test, acc_lines} ->
             {:cont, {test, [line | acc_lines]}}
         end,
         fn {previous_test, previous_test_lines} ->
           {:cont, {previous_test, Enum.reverse(previous_test_lines)}, {nil, []}}
         end
       )
    |> Stream.drop(1)
    |> Stream.map(fn {test, lines} ->
      case classify(lines) do
        {:ok, classification} ->
          {test, classification}

        :error ->
          Mix.shell().error("Could not classify #{test} error:\n#{Enum.join(lines)}")
          System.halt(1)
      end
    end)
    |> Enum.sort()
    |> Enum.each(fn {test, classification} ->
      Mix.shell().info("#{test}\t#{classification}")
    end)
  end

  defp classify([]) do
    :error
  end

  defp classify([" right: `SourceId" <> _ = line | tail]) do
    if String.contains?(line, "source spans cannot start and end in different files") do
      {:ok, "source spans cannot start and end in different files"}
    else
      classify(tail)
    end
  end

  defp classify([~S|Assertion failed: (*this && "isa<> used on a null type.")| <> _ | _]) do
    {:ok, "isa<> used on a null type"}
  end

  defp classify([
    ~S|Assertion failed: (funcOp.isExternal() && "cannot define a function more than once"| <>
    _
    | _
  ]) do
    {:ok, "cannot define a function more than once"}
  end

  defp classify(["Assertion failed: (hasVal), function getValue, " <> suffix | tail]) do
    if String.contains?(suffix, "llvm/ADT/Optional.h") do
      {:ok, "optional does not have value"}
    else
      classify(tail)
    end
  end

  defp classify(["error: attribute is already defined\n" | _]) do
    {:ok, "attribute is already defined"}
  end

  defp classify(["error: could not find file\n" | _]) do
    {:ok, "could not find file"}
  end

  defp classify(["error: could not resolve variable\n" | _]) do
    {:ok, "could not resolve variable"}
  end

  defp classify(["error: 'eir.binary.match.integer' op attribute 'unit' failed to satisfy constraint: 64-bit signless integer attribute\n" | _]) do
    {:ok, "'eir.binary.match.integer' op attribute 'unit' failed to satisfy constraint: 64-bit signless integer attribute"}
  end

  defp classify([
    "error: 'eir.logical.and' op result #0 must be 1-bit signless integer, but got '!eir.bool'\n"
    | _
  ]) do
    {:ok, "'eir.logical.and' op result must be 1-bit signless integer, but got '!eir.bool'"}
  end

  defp classify([
    "error: 'eir.logical.or' op result #0 must be 1-bit signless integer, but got '!eir.bool'\n"
    | _
  ]) do
    {:ok, "'eir.logical.or' op result must be 1-bit signless integer, but got '!eir.bool'"}
  end

  defp classify([
    "error: 'eir.map.contains' op operand #1 must be dynamic term type, but got '!eir.atom'\n"
    | _
  ]) do
    {:ok, "'eir.map.contains' op operand #1 must be dynamic term type"}
  end

  defp classify([
    <<"error: 'eir.map.insert' op operand #", _, " must be dynamic term type">> <> _ | _
  ]) do
    {:ok, "'eir.map.insert' op operand #0 must be dynamic term type"}
  end

  defp classify(["error: 'eir.map.update' op operand #0 must be dynamic term type, but got '!eir.box<!eir.map>'\n" | _]) do
    {:ok, "'eir.map.update' op operand #0 must be dynamic term type"}
  end

  defp classify([
    "error: 'eir.throw' op operand #2 must be opaque trace reference, but got '!eir.term'\n"
    | _
  ]) do
    {:ok, "'eir.throw' op operand #2 must be opaque trace reference, but got '!eir.term'"}
  end

  defp classify(["error: invalid cast type, source type is unsupported\n" | _]) do
    {:ok, "invalid cast type, source type is unsupported"}
  end

  defp classify(["error: invalid const expression\n" | _]) do
    {:ok, "invalid const expression"}
  end

  defp classify(["error: invalid string escape\n" | _]) do
    {:ok, "invalid string escape"}
  end

  defp classify(["error: invalid tuple type element" <> _ | _]) do
    {:ok, "invalid tuple type element"}
  end

  defp classify([<<"error: operand #", _, " does not dominate this use">> <> _ | _]) do
    {:ok, "operand does not dominate this use"}
  end

  defp classify(["error: operand type '!eir.nil' and result type '!eir.box<!eir.cons>' are not cast compatible\n" | _]) do
    {:ok, "operand type '!eir.nil' and result type '!eir.box<!eir.cons>' are not cast compatible"}
  end

  defp classify(["error: redefinition of symbol" <> _ | _]) do
    {:ok, "redefinition of symbol"}
  end

  defp classify(["error: undefined macro\n" | _]) do
    {:ok, "undefined macro"}
  end

  defp classify(["error: unrecognized token\n" | _]) do
    {:ok, "unrecognized token"}
  end

  defp classify(["stderr: error: invalid input file" <> _ | _]) do
    {:ok, "invalid input file"}
  end

  defp classify(["thread '<unknown>' has overflowed its stack\n" | _]) do
    {:ok, "stack overflow"}
  end

  defp classify(["thread '<unnamed>' panicked at 'expected constant const_value" <> suffix | tail]) do
    [_, after_number] = String.split(suffix, " ", parts: 2)

    case after_number do
      "to be an atom'" <> _ -> {:ok, "expected constant to be an atom"}
      _ -> classify(tail)
    end
  end

  defp classify(["thread '<unnamed>' panicked at 'expected primop value'" <> _ | _]) do
    {:ok, "expected primop value"}
  end

  defp classify(["thread '<unnamed>' panicked at 'expected value, but got pseudo-value'" <> _ | _]) do
    {:ok, "expected value, but go pseudo-value"}
  end

  defp classify([
    "thread '<unnamed>' panicked at 'invalid operation kind: UnpackValueList" <> _ | _
  ]) do
    {:ok, "invalid operation kind: UnpackValueList"}
  end

  defp classify(["thread '<unnamed>' panicked at 'no entry found for key', " <> suffix | tail]) do
    if String.ends_with?(suffix, "libeir_syntax_erl/src/lower/expr/record.rs:138:19\n") do
      {:ok, "no entry found for key in record"}
    else
      classify(tail)
    end
  end

  defp classify(["thread '<unnamed>' panicked at 'not implemented', " <> suffix | tail]) do
    if String.ends_with?(suffix, "libeir_syntax_erl/src/lower/expr/comprehension.rs:130:44\n") do
      {:ok, "binary generators not implemented"}
    else
      classify(tail)
    end
  end

  defp classify([
    "thread '<unnamed>' panicked at 'not yet implemented: unimplemented call type LocalDynamic" <>
    _
    | _
  ]) do
    {:ok, "unimplemented call type LocalDynamic"}
  end

  defp classify(["thread '<unnamed>' panicked at 'the given value is not a known block" <> _ | _]) do
    {:ok, "the given value is not a known block"}
  end

  defp classify(["thread 'lumen::otp::" <> suffix | tail]) do
    if String.contains?(suffix, "Compilation timed out") do
      {:ok, "compilation timed out"}
    else
      classify(tail)
    end
  end

  defp classify(["Undefined symbols" <> _ | _]) do
    {:ok, "undefined symbols"}
  end

  defp classify(["warning: invalid compile option\n" | _]) do
    {:ok, "invalid compile option"}
  end

  defp classify(["stdout: TODO file directive -file" <> _ | _]) do
    {:ok, "file directive (-file) unimplemented"}
  end

  defp classify([_ | tail]), do: classify(tail)
end
