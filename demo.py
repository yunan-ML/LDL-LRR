from scipy.io import loadmat
from ldllrr import LDL_LRR
from utils import report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# load dataset
data = loadmat('datasets/sj.mat')
X, D = data['features'], data['label_distribution']
Xr, Xs, Dr, Ds = train_test_split(X, D, test_size=0.1, random_state=0)
scaler = MinMaxScaler().fit(Xr)
Xr, Xs = scaler.transform(Xr), scaler.transform(Xs)

# train GLEMR
lrr = LDL_LRR(lam=1e-2, beta=1).fit(Xr, Dr)
Dhat = lrr.predict(Xs)

# report the results
report(Dhat, Ds, ds='sj')