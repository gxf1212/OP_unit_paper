# Molecular-dynamics input record

This directory records the study-specific PMF settings selected from the
`with-membrane/amber` calculation archive. The `.mdp` files describe the
minimization, staged equilibration, umbrella-window equilibration and 10 ns
production settings.

`example_C003/` is a representative input snapshot of a carboxylate 3-mer
comprising three methacrylate-backbone repeat units. It contains the topology,
index file and equilibrated coordinate snapshots. The main reference dataset
uses standardized truncated models of the local side-chain/headgroup
environment; the complete head-model systems are part of the institutional HPC
archive and are not reproduced here.

The example topology retains the original `amber14sb_OL15.ff/*` include names,
including AMBER/Lipid14 files generated with CHARMM-GUI FF-Converter. These
third-party dependencies are not redistributed here and must be obtained from
their documented sources before attempting a GROMACS run. Production
trajectories, binary run files, failed runs and cluster-specific SLURM scripts
are intentionally excluded.
