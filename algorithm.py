import numpy as np
import pandas as pd

class BayesianClassifier:

  def __init__(self, dataframe, num_classes):
    self.__dataframe = dataframe
    self.num_classes = num_classes
    self.__dataframe_divided = []
    self.__names_clases = []
  
  def mesures(self, verbose=False):

    # divide the dataframe by class

    last_column = self.__dataframe.columns[-1]
    for _, row in self.__dataframe.iterrows():

      if(len(self.__dataframe_divided) >= self.num_classes): break

      value = row[last_column]
      if(value not in self.__names_clases):
        self.__names_clases.append(value)
        self.__dataframe_divided.append({'name': value, 'measures': [], 'data': []})

    for i in range(self.num_classes):
      for _, row in self.__dataframe.iterrows():
        if row.iloc[-1] == self.__dataframe_divided[i]['name']:
          row_without_class = row.iloc[:-1]
          self.__dataframe_divided[i]['data'].append(row_without_class.tolist())

    # mesaures
          
    for i in self.__dataframe_divided:


      temp_df = pd.DataFrame(i['data'])
      var_for_each_col = temp_df.var()

      for j in range(self.__dataframe.shape[1] - 1):
        sum = 0
        for k in i['data']:
          sum += k[j]

        i['measures'].append({
          'column': "{}".format(self.__dataframe.columns.tolist()[j]),
          'mean': sum / len(i['data']),
          'var': var_for_each_col[j]
        })
      

    if verbose: print("\nDataframe divided: ", self.__dataframe_divided, "\n")
      

  def predictOne(self, y):
    probabilities_for_each_class = []

    for i in self.__dataframe_divided:
      probabilities_for_each_class.append(len(i['data']) / self.__dataframe.shape[0])

    final_probabilities = [[] for _ in range(len(probabilities_for_each_class))]

    for i, object in enumerate(self.__dataframe_divided):
        
      for j, measure in enumerate(object['measures']):
        x = y[j]
        final_probabilities[i].append({
          'probability': "P({} | {})".format(object['name'], measure['column']),
          'value': (1 / np.sqrt(2 * np.pi * measure['var'])) * np.exp((-(x - measure['mean']) ** 2) / (2 * measure['var']))
        })

    predictions = []
    for index, value in enumerate(probabilities_for_each_class):
      
      res = value
      for i in final_probabilities[index]:
        res *= i['value']

      predictions.append(res)

    maxValue = 0
    for i in range(1, len(predictions)):
      if predictions[i] > predictions[maxValue]:
        maxValue = i

    print("prediction: {}".format(self.__names_clases[maxValue]))

  def predictMany(self, y):
    for pred in y:
      self.predictOne(y = pred)
