#!/bin/bash

filename="dev1.zip"

wget -q -P "./tests/.data/" "http://www.irisa.fr/metiss/SiSEC10/underdetermined/${filename}"
unzip -d "./tests/.data/" "./tests/.data/${filename}"
