;;; vx-mode-tests.el --- Tests for vx-mode -*- lexical-binding: t; -*-

;; Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
;; See LICENSE for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

;;; Commentary:

;; Run from the repository root with:
;;
;;   emacs --batch -L emacs-vx -l emacs-vx/vx-mode-tests.el -f ert-run-tests-batch-and-exit

;;; Code:

(require 'ert)
(require 'vx-mode)

(defconst vx-test--repo-root
  (expand-file-name ".." (file-name-directory (or load-file-name buffer-file-name)))
  "The Vx repository root, one level above this file.")

(defun vx-test--face-at (code text)
  "Highlight CODE and return the face of the first occurrence of TEXT in it."
  (with-temp-buffer
    (vx-mode)
    (insert code)
    (font-lock-ensure)
    (goto-char (point-min))
    (search-forward text)
    (get-text-property (match-beginning 0) 'face)))

(ert-deftest vx-keywords-are-highlighted ()
  (dolist (word '("fn" "let" "where" "type" "spawn" "comptime" "macro_rules"
                "pub" "transfer_async"))
    (should (eq (vx-test--face-at (concat word " x") word)
                'font-lock-keyword-face))))

(ert-deftest vx-keyword-inside-a-longer-name-is-not-highlighted ()
  (should (eq (vx-test--face-at "let fn_name = 1;" "fn_name")
              'font-lock-variable-name-face))
  (should-not (vx-test--face-at "let a = typed;" "typed")))

(ert-deftest vx-type-names ()
  (should (eq (vx-test--face-at "let x : i64 = 0;" "i64") 'font-lock-type-face))
  (should (eq (vx-test--face-at "let x : bf16 = y;" "bf16") 'font-lock-type-face))
  (should (eq (vx-test--face-at "let o : Option<T> = y;" "Option") 'font-lock-type-face))
  (should (eq (vx-test--face-at "Memory HBM {}" "Memory") 'font-lock-builtin-face)))

(ert-deftest vx-all-caps-names-are-constants ()
  (should (eq (vx-test--face-at "transfer(a, Memory::NPU_HBM);" "NPU_HBM")
              'font-lock-constant-face)))

(ert-deftest vx-definition-names ()
  (should (eq (vx-test--face-at "fn add_one(x : i64) -> i64 {" "add_one")
              'font-lock-function-name-face))
  (should (eq (vx-test--face-at "struct point {" "point") 'font-lock-type-face))
  (should (eq (vx-test--face-at "macro_rules stamp_from {" "stamp_from")
              'font-lock-function-name-face))
  (should (eq (vx-test--face-at "let mut total = 0;" "total")
              'font-lock-variable-name-face)))

(ert-deftest vx-line-comment-and-doc-comment ()
  (should (eq (vx-test--face-at "// plain fn\nfn f() {}" "plain")
              'font-lock-comment-face))
  (should (eq (vx-test--face-at "/// The sum.\nfn f() {}" "The sum")
              'font-lock-doc-face))
  ;; The lexer treats four slashes as a plain comment, not a doc comment.
  (should (eq (vx-test--face-at "//// banner\nfn f() {}" "banner")
              'font-lock-comment-face))
  ;; The comment ends at the end of the line.
  (should (eq (vx-test--face-at "// note\nfn f() {}" "fn")
              'font-lock-keyword-face)))

(ert-deftest vx-string-with-escaped-quote ()
  (let ((code "print!(\"say \\\"fn\\\" here\"); let x = 1;"))
    ;; The escaped quote does not end the string, so `fn' is part of it.
    (should (eq (vx-test--face-at code "fn") 'font-lock-string-face))
    (should (eq (vx-test--face-at code "let") 'font-lock-keyword-face))))

(ert-deftest vx-macro-call-but-not-not-equal ()
  (should (eq (vx-test--face-at "print!(\"hi\");" "print!")
              'font-lock-preprocessor-face))
  (should-not (vx-test--face-at "if a!=b {}" "a")))

(ert-deftest vx-numbers ()
  (dolist (number '("42" "1_000" "0x9E37" "1.5e-3" "2.0f32" "7i64"))
    (should (eq (vx-test--face-at (concat "let x = " number ";") number)
                vx--number-face)))
  ;; In `0..n' only the 0 is a number, the `..' is a range.
  (should-not (vx-test--face-at "for i in 0..n {}" "..")))

(defun vx-test--reindent (text)
  "Remove all indentation from TEXT, indent it with `vx-mode' and return it."
  (with-temp-buffer
    (vx-mode)
    (insert text)
    (goto-char (point-min))
    (while (re-search-forward "^[ \t]+" nil t)
      (replace-match ""))
    (indent-region (point-min) (point-max))
    (buffer-string)))

(ert-deftest vx-indentation-matches-vx-format ()
  "Files in stdlib/ are already formatted by vx-format.
Removing their indentation and indenting them again must give back the same
text.  tensor.vx has calls split over several lines, whose arguments are
not indented further."
  (dolist (file '("stdlib/core/clone.vx" "stdlib/core/cmp.vx"
                  "stdlib/core/convert.vx" "stdlib/core/hash.vx"
                  "stdlib/std/tensor.vx"))
    (let ((original (with-temp-buffer
                      (insert-file-contents (expand-file-name file vx-test--repo-root))
                      (buffer-string))))
      (should (equal (cons file (vx-test--reindent original))
                     (cons file original))))))

(ert-deftest vx-closing-brace-is-indented-when-typed ()
  (with-temp-buffer
    (vx-mode)
    (insert "fn f() {\n  return 1;\n  ")
    (let ((last-command-event ?}))
      (self-insert-command 1)
      (electric-indent-post-self-insert-function))
    (should (equal (buffer-string) "fn f() {\n  return 1;\n}"))))

(ert-deftest vx-mode-is-used-for-vx-files ()
  (should (eq (assoc-default "main.vx" auto-mode-alist #'string-match) 'vx-mode)))

;;; vx-mode-tests.el ends here
