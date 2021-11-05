import pandas as pd
import numpy as np

def convert_confidences(confidences):
    pass

def summarize_stats(df):
    # these never change so I'm hardcoding them
    # I'm too stubborn to admit I made a mistake while recording data so I'm just rolling with it
    fps = df.iloc[0].values[1:].astype(np.float32)
    inf_times = df.iloc[1].values[1:].astype(np.float32)
    # these are supposed to be tensors goddamnit
    confidences = df.iloc[2].values[1:]
    print(fps[:5])
    print(fps.mean())
    print(inf_times[:5])
    print(inf_times.mean())
    print(confidences[:5])


def main():
    fname = "../uncal_test.pkl"
    df = pd.read_pickle(fname)
    summarize_stats(df)

if __name__ == "__main__":
    main()