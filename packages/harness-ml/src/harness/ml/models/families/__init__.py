import os
import sys

# Prevent OMP/OpenMP segfault when multiple libraries with conflicting OpenMP
# runtimes (e.g. catboost + pytabkit/pytorch-lightning) are loaded together.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# On macOS, limit OMP threads to avoid pthread_mutex_init failures when
# catboost and pytorch-lightning coexist in the same process.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
