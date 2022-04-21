@echo off

set root=tests\.data
set filename=dev1.zip

mkdir %root%
bitsadmin /transfer "Download" http://www.irisa.fr/metiss/SiSEC10/underdetermined/%filename% %CD%\%root%\%filename%
call powershell -command "Expand-Archive %root%\%filename%"
dir dev1