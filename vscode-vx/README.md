# Vx Language Extension for VSCode

This is the official syntax highlighting extension for the Vx systems programming language.

## To install the extension from vsix file

### Using CLI
You can install the generated `.vsix` file in VSCode by running:
`code --install-extension vx-lang-X.X.X.vsix`

### Using the VS Code Interface (Easiest)
- Open Visual Studio Code.
- Click on the Extensions icon in the Activity Bar on the side (or press Ctrl+Shift+X on Windows/Linux, Cmd+Shift+X on macOS).
- Click the Views and More Actions (the three dots ...) icon at the top right of the Extensions view.
- Select Install from VSIX... from the dropdown menu.
- Browse to and select your .vsix file, then click Install.
- A notification will appear once the installation is complete; you may need to restart VS Code to activate it.
- If you have an older version installed, you might need to remove it if vscode couldn't disambiguate

## For Developers

To extend or modify this plugin:

1. **Modify the Syntax**: Edit `syntaxes/vx.tmLanguage.json` to add new keywords, types, or syntax rules.
2. **Modify the Configuration**: Edit `language-configuration.json` for commenting rules, bracket matching, and auto-closing pairs.
3. **Packaging**: To build a new `.vsix` release, make sure you have `vsce` installed (`npm install -g @vscode/vsce`).
4. **Build**: Run `vsce package` in this directory to generate a `.vsix` file.


