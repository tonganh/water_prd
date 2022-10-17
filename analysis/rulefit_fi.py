import numpy as np
import pandas as pd
from analysis.rulefit import RuleFit
from sklearn.ensemble import GradientBoostingRegressor


data_df = pd.read_csv("data/hoat_chat.csv")
data_df = data_df.drop(columns=["Date","Site","Sample", "HAAFP"])
data_df = data_df.dropna()
y = data_df.THMFP.values
X = data_df.drop("THMFP", axis=1)
features = X.columns
X = X.to_numpy()


# gb = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.01)
# rf = RuleFit(gb)
rf = RuleFit()

rf.fit(X, y, feature_names=features)

rf.predict(X)

rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
import pdb
pdb.set_trace()

print(rules)