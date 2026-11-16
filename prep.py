import sys, pickle, pandas as pd
from common import make_split
df = pd.read_pickle(sys.argv[1])
for name in ("B", "A", "final"):
    out = make_split(df, name)
    pickle.dump(out, open(f"split_{name}.pkl", "wb"))
    print(name, out[0].shape, out[2].shape)
