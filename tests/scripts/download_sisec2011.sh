#!/bin/bash

root="./tests/.data"
filename="dev1.zip"

wget -q -P "${root}/" "http://www.irisa.fr/metiss/SiSEC10/underdetermined/${filename}"
unzip -q -d "${root}/" "${root}/${filename}"
