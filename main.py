import pandas as pd
from algorithm import BayesianClassifier

df = pd.read_csv("./data/Raisin_Dataset.csv")

column_names = df.columns
y = []

for i in range(len(column_names) - 1):
  value = input("Enter the value for " + column_names[i] + " : ")
  y.append(float(value))

verboseChoice = input("Would you like to view the process in the console (yes/no)? : ")  

verbose = True if verboseChoice == "yes" else False

model = BayesianClassifier(dataframe= df, num_classes= 2)

model.mesures(verbose=verbose)


model.predictOne(y= y)
