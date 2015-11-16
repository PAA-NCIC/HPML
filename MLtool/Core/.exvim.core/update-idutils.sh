#!/bin/bash
export DEST="./.exvim.core"
export TOOLS="/home/ylf/.vim/tools/"
export TMP="${DEST}/_ID"
export TARGET="${DEST}/ID"
sh ${TOOLS}/shell/bash/update-idutils.sh
