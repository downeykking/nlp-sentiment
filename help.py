import pandas as pd

df1 = pd.read_csv("./train.csv")
df2 = pd.read_csv("./train.csv")
df1 = pd.read_csv("./train.csv")


def fun(x):
    if x == 0:
        return "A"
    elif x == 1:
        return "B"
    elif x == 2:
        return "C"
    elif x == 3:
        return "D"
    else:
        return "E"


def help(path, name):
    df = pd.read_csv(path)
    df['Sentiment'] = df['Sentiment'].apply(func=fun)
    df.to_csv(name, index=False)


help("train.csv", "1.csv")
help("val.csv", "2.csv")
