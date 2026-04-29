# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODEL_OUTPUT = Path(
    r"C:\Users\maple\Documents\Academia\UW\SPHyNX_Lab\Model output"
)
# %%
fn = MODEL_OUTPUT / "cx_p05_wec05_d2_wmd025_sg05_t30" \
                  / "c10_p05_wec05_d2_wmd025_sg05_t30" \
                  / "data" \
                  / "objectforces.txt"

print(fn)

# %%
df = pd.read_csv(fn, sep=r"\s+")
df.head()

# %%
plt.plot(df.iloc[:,0], df.iloc[:,1])
plt.xlabel("column 0")
plt.ylabel("column 1")
plt.show()
# %%
