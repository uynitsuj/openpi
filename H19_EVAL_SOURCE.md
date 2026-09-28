# H19 evaluation source tarball

`h19_eval_setup.sh` unpacks `h19_abc_source_eab8e9b3.tar.gz` (abc-rabc source at commit `eab8e9b3`) and checks its digest:

- sha256 `e14871964779fd0a381e86dd0b0444b078aa860ef12ce2a13b5ce0d40b810048`
- stored at `s3://xdof-internal-research/new_sim_curation_20260904/h19-base-bottles-v1/code/h19_abc_source_eab8e9b3.tar.gz`

The tarball is kept out of git (binary, 14 MB); copy it next to `h19_eval_setup.sh` before running the eval setup.
