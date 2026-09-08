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
