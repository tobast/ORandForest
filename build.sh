#!/bin/bash

#set -x # DEBUG

rm -rf _build
dune build @install
dune install
