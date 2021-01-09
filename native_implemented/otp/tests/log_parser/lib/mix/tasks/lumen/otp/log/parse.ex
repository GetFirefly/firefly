defmodule Mix.Tasks.Lumen.Otp.Log.Parse do
  @moduledoc "Parses the test.log outputted from `cargo test --package liblumen_otp -- lumen::otp`"
  @shortdoc "Parses test.log"

  alias Lumen.OTP.Log.Parser

  use Mix.Task

  @impl Mix.Task

  def run([path]) do
    Parser.parse(path)
  end

  def run(_) do
    Mix.shell().info("mix lumen.otp.log.parse LOG_PATH")
  end
end
