Draft for the `mceq-project/mceq-examples` README (maintainer action — this
repo's CI and files are not affected; archived here until the GitHub repo is
updated/archived):

---

# MCEq examples

This repository is **archived**. Its notebooks were folded into the MCEq
repository and are executed by CI there:

- sources: <https://github.com/mceq-project/MCEq/tree/main/examples>, as
  marimo notebooks that also export to Jupyter; rendered pages are linked
  from <https://mceq.readthedocs.io/en/latest/example-gallery.html>
- every notebook runs on MCEq 2.x with the v2 release databases
- `DDM_example.ipynb` and `Modify_particle_yields.ipynb` were unique to this
  repository and are now `examples/hadronic/ddm.py` and
  `examples/hadronic/modify_particle_yields.py`;
  `work_in_progress/contribution_from_individual_hadrons.ipynb` is covered by
  `examples/hadronic/partial_hadron_*.py`; `work_in_progress/DDM_vs_daemonflux.ipynb`
  depended on a private fit file and was dropped

Open an issue in the MCEq repository if a tutorial you relied on is missing.

---
