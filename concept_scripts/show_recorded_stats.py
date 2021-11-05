import pandas as pd
import numpy as np
import cv2

def convert_confidences(confidences):
    """
    Converts a list of tensors to a single np array
    :param confidences:
    :return:
    """
    lst = []
    for tensor in confidences:
        for val in tensor:
            lst.append(val.numpy())
    return np.array(lst)

def summarize_stats(df):
    # these never change so I'm hardcoding them
    # the 1: is to index only the values, and skip the row name
    fps = df.iloc[0].values[1:].astype(np.float32)
    inf_times = df.iloc[1].values[1:].astype(np.float32)
    confidences = df.iloc[2].values[1:]
    confidences = convert_confidences(confidences)
    avg_fps = fps.mean()
    avg_inf_time = inf_times.mean()
    avg_conf = confidences.mean()
    data = np.array([[avg_fps], [avg_inf_time], [avg_conf]]).T
    cols = ['Average FPS', 'Average Inference Time', 'Average Confidence']
    df = pd.DataFrame(data, columns=cols)
    return df

def show_imgs(df, time=1000):
    imgs = df.iloc[3]
    imgs = imgs.dropna()
    for img in imgs:
        cv2.imshow("IMG", img)
        cv2.waitKey(time)
    cv2.destroyAllWindows()

def main():
    fname = "../uncal_test.pkl"
    df = pd.read_pickle(fname)
    stats = summarize_stats(df)
    print(stats.to_markdown())
    show_imgs(df, time=500)

if __name__ == "__main__":
    main()