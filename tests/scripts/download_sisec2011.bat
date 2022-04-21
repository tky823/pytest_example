@echo off

set root=tests\.data
set filename=dev1.zip

mkdir %root%
bitsadmin /transfer "Download" http://www.irisa.fr/metiss/SiSEC10/underdetermined/dev1.zip %CD%\%root%\%filename%
dir %root%