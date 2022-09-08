For each setting, run 20 times:
    * HPOLib (4 datasets) with the knowledge transfer from 3 datasets
    * NMTBench (3 datasets) with the knowledge transfer from 3 datasets

(10) Optimizers:
    * (1) Meta-learn TPE (100 random samplings from each source + 100 observations)
    * (1) TPE with uniform transfer (100 random samplings from each source + 100 observations)
    * (1) TPE (100 observations)
    * (4) Meta-learn BO (100 random samplings from each source + 100 observations)
    * (2) BO (100 observations)
    * (1) Random search (100 observations)

NOTE: BO means ParEGO and EHVI and meta-learn BO means RGPE and TSTR.

10 optimizers x 7 datasets x 20 runs = 1400 runs
