from sklearn.metrics import accuracy_score,precision_score,recall_score
from pandas import pd

df = pd.read_csv(r"G:\python\ML\sklearn\.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values