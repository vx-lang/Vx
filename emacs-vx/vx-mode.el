;;; vx-mode.el --- Major mode for the Vx programming language -*- lexical-binding: t; -*-

;; Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
;; See LICENSE for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

;; Keywords: languages
;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; Syntax highlighting, indentation and imenu support for `.vx' files.
;;
;; The keyword lists follow `KEYWORDS' in src/lexer.rs.  When a keyword is
;; added to the lexer, add it here too.
;;
;; To use it, put this in your init file:
;;
;;   (add-to-list 'load-path "/path/to/Vx/emacs-vx")
;;   (require 'vx-mode)

;;; Code:

(defgroup vx nil
  "Support for the Vx programming language."
  :group 'languages
  :prefix "vx-")

(defcustom vx-indent-offset 2
  "Number of spaces for each level of `{ }' nesting.
The default matches the output of `vx-format'."
  :type 'integer
  :safe #'integerp)

(defconst vx-keywords
  '("across" "as" "assert" "break" "comptime" "continue" "else" "ensures"
    "enum" "extern" "fn" "for" "grad" "if" "impl" "import" "in" "invariant"
    "jvp" "let" "loop" "macro_rules" "match" "mut" "on" "requires" "return"
    "safe" "self" "spawn" "struct" "tasks" "trait" "transfer" "type" "unroll"
    "unsafe" "vjp" "where" "while")
  "Keywords of Vx, as listed in src/lexer.rs, plus `self'.")

(defconst vx-builtin-types
  '("Topology" "Memory" "Ref" "Verified" "Pinned" "HardwareState")
  "Type names that the Vx lexer treats as keywords.")

(defconst vx-primitive-types
  '("i4" "i8" "i16" "i32" "i64" "i128"
    "u4" "u8" "u16" "u32" "u64" "u128"
    "f16" "bf16" "f32" "f64" "bool" "usize" "isize")
  "Built-in number and boolean types.")

(defconst vx-constants '("true" "false")
  "Literal constants.")

(defconst vx--ident "[[:alpha:]_][[:alnum:]_]*"
  "Regexp for an identifier.")

(defvar vx-mode-syntax-table
  (let ((table (make-syntax-table)))
    ;; `//' starts a comment that ends at the end of the line.  Vx has no
    ;; block comments.
    (modify-syntax-entry ?/ ". 12" table)
    (modify-syntax-entry ?\n ">" table)
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\\ "\\" table)
    (modify-syntax-entry ?_ "_" table)
    ;; These are operators, not brackets.
    (dolist (c '(?< ?> ?+ ?- ?* ?% ?& ?| ?^ ?= ?! ?@ ?$ ?? ?'))
      (modify-syntax-entry c "." table))
    table)
  "Syntax table for `vx-mode'.")

(defun vx--doc-comment-p (start)
  "Return non-nil if the comment at START is a `///' doc comment.
Like the lexer, four or more slashes make a normal comment."
  (save-excursion
    (goto-char start)
    (and (looking-at-p "///") (not (looking-at-p "////")))))

(defun vx--syntactic-face (state)
  "Pick the face for the string or comment described by STATE."
  (cond ((nth 3 state) 'font-lock-string-face)
        ((vx--doc-comment-p (nth 8 state)) 'font-lock-doc-face)
        (t 'font-lock-comment-face)))

(defconst vx--number-face
  (if (facep 'font-lock-number-face) 'font-lock-number-face 'font-lock-constant-face)
  "Face for number literals.  `font-lock-number-face' exists from Emacs 29.")

(defconst vx-font-lock-keywords
  `(;; `macro_rules name' and `macro_rules! name'.
    (,(concat "\\_<macro_rules!?[[:space:]]+\\(" vx--ident "\\)")
     1 font-lock-function-name-face)
    ;; The name in `fn name'.
    (,(concat "\\_<fn[[:space:]]+\\(" vx--ident "\\)")
     1 font-lock-function-name-face)
    ;; The name in `struct Name', `enum Name', `trait Name', `type Name'.
    (,(concat "\\_<\\(?:struct\\|enum\\|trait\\|type\\)[[:space:]]+\\("
              vx--ident "\\)")
     1 font-lock-type-face)
    ;; A macro call such as `print!(...)'.  `!=' is an operator, not a macro.
    (,(concat "\\_<\\(" vx--ident "!\\)\\(?:[^=]\\|$\\)")
     1 font-lock-preprocessor-face)
    (,(regexp-opt vx-keywords 'symbols) . font-lock-keyword-face)
    (,(regexp-opt vx-constants 'symbols) . font-lock-constant-face)
    (,(regexp-opt vx-builtin-types 'symbols) . font-lock-builtin-face)
    (,(regexp-opt vx-primitive-types 'symbols) . font-lock-type-face)
    ;; A macro variable such as `$t' inside `macro_rules'.
    (,(concat "\\$" vx--ident) . font-lock-variable-name-face)
    ;; The names bound by `let x', `let mut x' and `for x in'.
    (,(concat "\\_<\\(?:let\\|for\\)[[:space:]]+\\(?:mut[[:space:]]+\\)?\\("
              vx--ident "\\)")
     1 font-lock-variable-name-face)
    ;; Numbers: 42, 1_000, 0x9E37, 1.5e-3, 2.0f32, 7i64.
    ("\\_<[0-9][0-9_]*\\(?:\\.[0-9][0-9_]*\\)?\\(?:[eE][-+]?[0-9_]+\\)?[[:alnum:]_]*"
     . ',vx--number-face)
    ;; ALL_CAPS names, such as `Memory::HBM', are constants.
    ("\\_<[A-Z][A-Z0-9_]+\\_>" . font-lock-constant-face)
    ;; Other capitalized names, such as `Option' or `T', are types.
    ("\\_<[A-Z][[:alnum:]_]*\\_>" . font-lock-type-face))
  "Highlighting rules for `vx-mode'.  The first rule that matches wins.")

(defun vx--brace-depth (pos)
  "Return how many `{' are open at POS, outside strings and comments.
Open `(' and `[' do not count: `vx-format' does not indent for them."
  (let ((count 0))
    (dolist (open (nth 9 (syntax-ppss pos)) count)
      (when (eq (char-after open) ?{)
        (setq count (1+ count))))))

(defun vx-indent-line ()
  "Indent the current line the way `vx-format' does."
  (interactive)
  (let ((ppss (save-excursion (syntax-ppss (line-beginning-position)))))
    ;; Leave the inside of a multi-line string alone.
    (unless (nth 3 ppss)
      (let* ((depth (save-excursion
                      (back-to-indentation)
                      (vx--brace-depth (point))))
             (closes (save-excursion
                       (back-to-indentation)
                       (looking-at-p "}")))
             (column (* vx-indent-offset (max 0 (if closes (1- depth) depth))))
             (offset (- (current-column) (current-indentation))))
        (indent-line-to column)
        ;; Keep point where it was relative to the text.
        (when (> offset 0)
          (forward-char offset))))))

(defconst vx-imenu-generic-expression
  `(("Functions" ,(concat "^[[:space:]]*\\(?:unsafe[[:space:]]+\\)?fn[[:space:]]+\\("
                          vx--ident "\\)")
     1)
    ("Types" ,(concat "^[[:space:]]*\\(?:struct\\|enum\\|trait\\)[[:space:]]+\\("
                      vx--ident "\\)")
     1)
    ("Macros" ,(concat "^[[:space:]]*macro_rules!?[[:space:]]+\\(" vx--ident "\\)")
     1))
  "Rules that list functions, types and macros in the `imenu' menu.")

;;;###autoload
(define-derived-mode vx-mode prog-mode "Vx"
  "Major mode for editing Vx source code.

\\{vx-mode-map}"
  :syntax-table vx-mode-syntax-table
  (setq-local comment-start "// ")
  (setq-local comment-end "")
  (setq-local comment-start-skip "//+[[:space:]]*")
  (setq-local font-lock-defaults
              '(vx-font-lock-keywords
                nil nil nil nil
                (font-lock-syntactic-face-function . vx--syntactic-face)))
  (setq-local indent-line-function #'vx-indent-line)
  (setq-local indent-tabs-mode nil)
  (setq-local electric-indent-chars (cons ?} electric-indent-chars))
  (setq-local imenu-generic-expression vx-imenu-generic-expression))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.vx\\'" . vx-mode))

(provide 'vx-mode)

;;; vx-mode.el ends here
