#!/bin/bash
# bash bcc_ligand_nmer.sh P007 3 "CC(C)(C(=O)OCC[N+](CC)(CC)CCOP(=O)([O-])OC)"
f=$1
n=$2
mono=$3

mkdir ${f}-${n}
cd ${f}-${n}
poly=`printf "%0.s${mono}" $(seq $n)`
obabel -ismi -:${poly} -omol2 -O ${f}-${n}.mol2 --gen3D -p 7.4
obminimize -ff MMFF94 -n 1000 ${f}-${n}.mol2 > ${f}-${n}.pdb
acpype -i ${f}-${n}.pdb -n 0 -m 1 -a gaff -c bcc -o gmx
cp ${f}-${n}.acpype/${f}-${n}_GMX.itp ${f}-${n}.itp
gmx editconf -f ${f}-${n}.acpype/${f}-${n}_GMX.gro -o ${f}-${n}_center.pdb -center 0 0 0
sed -i 's/UNL/LIG/g' ${f}-${n}.itp
sed -i "s/${f}-${n}/LIG  /g" ${f}-${n}.itp
sed -i 's/UNL/LIG/g' ${f}-${n}_center.pdb
