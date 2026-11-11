# Editor support

## VS Code

The extension lives in `vscode-vx/` in the repository. It provides syntax highlighting and connects
to the language server.

Until it is published to the marketplace, install it from source:

```bash
cd vscode-vx
npm install
npm run package
code --install-extension vx-*.vsix
```

## Emacs

The major mode `vx-mode` lives in `emacs-vx/` in the repository. It highlights keywords, types,
numbers, strings and macro calls. It indents 2 spaces for each open `{`, as `vx-format` does. It
also gives `imenu` a list of the functions, types and macros of the file.

To install it, add these lines to your Emacs init file:

```elisp
(add-to-list 'load-path "/path/to/Vx/emacs-vx")
(require 'vx-mode)
```

Emacs then opens each `.vx` file in `vx-mode`. To change the indent width, set `vx-indent-offset`.

`vx-mode` does not start the language server. To use `vx-analyzer` with Eglot, the LSP client of
Emacs, add this line too:

```elisp
(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs '(vx-mode "vx-analyzer")))
```

## The language server

`vx-analyzer` speaks the Language Server Protocol, so any LSP-capable editor can use it. It ships
with the toolchain, at `~/.vx/bin/vx-analyzer` for a standard install.

Point your editor's LSP client at that binary for files with the `.vx` extension. In Neovim with
`nvim-lspconfig`, for example, that is a `cmd` of `{ "vx-analyzer" }` and a `filetypes` of
`{ "vx" }`.

## Formatting

`vx-format` is the canonical formatter. There is one style and no configuration:

```bash
vx-format path/to/file.vx
```

It rewrites files in place. Wire it to format-on-save in your editor, and run it over a directory
before committing — the project's CI checks that tracked `.vx` files are formatted.
