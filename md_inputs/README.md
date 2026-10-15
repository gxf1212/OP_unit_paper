# Molecular-dynamics input record

This directory records the study-specific PMF settings selected from the
`with-membrane/amber` calculation archive. The `.mdp` files describe the
minimization, staged equilibration, umbrella-window equilibration and 10 ns
production settings. The C003 example contains a representative topology,
index file and equilibrated coordinate snapshots.

The example topology retains the original `amber14sb_OL15.ff/*` include names.
Those force-field dependencies are not assumed to be supplied by this small
release snapshot; restore them from the documented force-field source before
attempting a GROMACS run. Production trajectories, binary run files, failed
runs and cluster-specific SLURM scripts are intentionally excluded.
