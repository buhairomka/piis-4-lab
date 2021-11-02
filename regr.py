from sklearn.linear_model import LinearRegression
import pandas

file = pandas.read_csv('res.csv', header=None)
parameters = file.drop(labels=3, axis=1)
predictions = file[3]
parameters.loc[parameters[1] == 'win', 1] = 1
parameters.loc[parameters[1] == 'lose', 1] = 0
parameters.loc[parameters[4] == 'ab', 4] = 0
parameters.loc[parameters[4] == 'emm', 4] = 1
model = LinearRegression()
model.fit(parameters, predictions)
forprediction=[
    [45, 1, 12.079936, 1],
    [46, 1, 12.079936, 1],
    [47, 1, 12.079936, 1],
    [48, 1, 12.079936, 1],
    [49, 1, 12.079936, 1],
    [50, 0, 0.5079936, 1],
]
for i in forprediction:
    print(i,model.predict([i]))