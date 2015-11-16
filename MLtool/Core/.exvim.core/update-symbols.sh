#!/bin/bash
export DEST="./.exvim.core"
export TOOLS="/home/ylf/.vim/tools/"
export TMP="${DEST}/_symbols"
export TARGET="${DEST}/symbols"
sh ${TOOLS}/shell/bash/update-symbols.sh
