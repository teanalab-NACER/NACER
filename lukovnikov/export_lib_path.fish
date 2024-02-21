#!/usr/bin/env fish

set PREFIX $HOME/miniconda3/envs/conv-er
set -gx LD_LIBRARY_PATH "$PREFIX/lib:$PREFIX/jre/lib:$LD_LIBRARY_PATH"
set -gx LD_LIBRARY_PATH "$PREFIX/jre/lib/amd64:$LD_LIBRARY_PATH"
set -gx LD_LIBRARY_PATH "$PREFIX/jre/lib/amd64/server:$LD_LIBRARY_PATH"
set -gx LD_LIBRARY_PATH "$PREFIX/lib/python2.7/site-packages:$LD_LIBRARY_PATH"