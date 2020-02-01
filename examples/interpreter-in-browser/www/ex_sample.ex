defmodule ExSample do
  # NOTE: this file is not automatically compiled to .erl, please check the
  # README.md file for more information.
  def run_me(arg) do
    :lumen_intrinsics.println("Doing the Lumen web work...")
    {:ok, window} = Lumen.Web.Window.window()
    {:ok, document} = Lumen.Web.Window.document(window)
    {:ok, paragraph} = Lumen.Web.Document.create_element(document, "p")

    text =
      Lumen.Web.Document.create_text_node(
        document,
        "This text was created through Elixir in your browser."
      )

    :ok = Lumen.Web.Node.append_child(paragraph, text)
    {:ok, output} = Lumen.Web.Document.get_element_by_id(document, "output")
    :ok = Lumen.Web.Node.append_child(output, paragraph)
    :lumen_intrinsics.println("Done!")
  end
end
