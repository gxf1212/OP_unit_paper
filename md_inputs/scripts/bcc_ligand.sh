#!/bin/bash
# bash bcc_ligand.sh hP005 "C[N+](CC)(CC)CCOP(=O)([O-])OC"
f=$1
smi=$2

mkdir ${f}
cd ${f}
obabel -ismi -:${smi} -omol2 -O ${f}.mol2 --gen3D -p 7.4
obminimize -ff MMFF94 -n 1000 ${f}.mol2 > ${f}_m.pdb
# bcc charge
acpype -i ${f}_m.pdb -n 0 -m 1 -a gaff -c bcc -o gmx
cp ${f}_m.acpype/${f}_m_GMX.itp ${f}.itp
gmx editconf -f ${f}_m.acpype/${f}_m_GMX.gro -o ${f}_center.pdb -center 0 0 0  # center at origin
sed -i 's/UNL/LIG/g' ${f}.itp
sed -i "s/${f}_m/LIG  /g" ${f}.itp
sed -i 's/UNL/LIG/g' ${f}_center.pdb
