#!/bin/bash
set -ev

sudo apt-get update -qq
sudo apt-get install -y gfortran libgsl0-dev

gfover=$(gfortran -dumpversion | cut -f1,2 -d.)
sudo ln -s /usr/lib/gcc/x86_64-linux-gnu/$gfover/libgfortran.so /usr/local/lib
