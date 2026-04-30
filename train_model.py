from Training_data import training

#import logistic regression
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

class AI():
    def making_the_model():

        model = LogisticRegression()
        vectorize = CountVectorizer()

        data = training()
        X = []
        Y = []

        for a,b in data:
            X.append(a)
            Y.append(b)

        vectorized_data = vectorize.fit_transform(X) 
        '''fit = makes the vocabulary of unique words
        transform = check how many times a word is present in the vocabulary every sentence (in the training data)'''

        trained_model = model.fit(vectorized_data, Y) #fits the vectorized data to the labels(Y_list) it them macthe like X[0] = Y[0]

        return trained_model, vectorize
