#!/bin/sh
# -*- mode: shell-script -*-
#
# Tangle .org files with org-mode, then normalise the output with black.
#
# The black pass is not cosmetic housekeeping: org-babel-tangle org-trims
# each block body and joins blocks with exactly one padding newline (see
# ob-tangle.el), so a file tangled from several blocks can never carry the
# two blank lines black requires before a top-level definition.  Formatting
# after tangling is the only way for the committed .py to be black-clean.
#
# `make check-tangle` runs this same script into a scratch directory, so the
# comparison stays exact.  black is a dev dependency; without it the tangled
# output will differ from what is committed and check-tangle will say so.
#
set -e
DIR=`pwd`
FILES=""

# wrap each argument in the code required to call tangle on it
for i in $@; do
    FILES="$FILES \"$i\""
done

emacs -Q --batch \
     --eval "(progn
     (require 'org)(require 'ob)(require 'ob-tangle)
     (mapc (lambda (file)
            (find-file (expand-file-name file \"$DIR\"))
            (org-babel-tangle)
            (kill-buffer)) '($FILES)))" 2>&1 |grep -i tangled

# Format exactly the files the tangle produced -- not the whole tree, so
# hand-written files alongside them are left alone.  BLACK is supplied by the
# Makefile (`poetry run black`); it falls back to a plain `black` on PATH.
targets=`sed -n 's/.*:tangle \([^ ]*\).*/\1/p' $@ | sort -u | grep -v '^no$'`
existing=""
for t in $targets; do
    [ -f "$t" ] && existing="$existing $t"
done

if [ -n "$existing" ]; then
    BLACK="${BLACK:-black}"
    if ! $BLACK -q $existing; then
        echo "tangle.sh: black failed or was not found (BLACK=$BLACK)." >&2
        echo "            The tangled output will not match what is committed;" >&2
        echo "            install the dev dependencies or set BLACK." >&2
        exit 1
    fi
fi
