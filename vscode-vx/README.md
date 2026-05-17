# Vx Language Extension for VSCode

This is the official syntax highlighting extension for the Vx systems programming language.

## For Developers

To extend or modify this plugin:

1. **Modify the Syntax**: Edit `syntaxes/vx.tmLanguage.json` to add new keywords, types, or syntax rules.
2. **Modify the Configuration**: Edit `language-configuration.json` for commenting rules, bracket matching, and auto-closing pairs.
3. **Packaging**: To build a new `.vsix` release, make sure you have `vsce` installed (`npm install -g @vscode/vsce`).
4. **Build**: Run `vsce package` in this directory to generate a `.vsix` file.

You can install the generated `.vsix` file in VSCode by running:
`code --install-extension vx-lang-X.X.X.vsix`
