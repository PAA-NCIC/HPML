#!/bin/bash
export DEST="./.exvim.fv"
export TOOLS="/home/ylf/.vim/tools/"
export IS_EXCLUDE=
export FOLDERS=""
export FILE_SUFFIXS=".*"
export TMP="${DEST}/_files"
export TARGET="${DEST}/files"
export ID_TARGET="${DEST}/idutils-files"
sh ${TOOLS}/shell/bash/update-filelist.sh
