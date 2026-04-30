from train import AI

import random

from sklearn.linear_model import LogisticRegression

from god_api import god

from motivational_quotes import motivation

from healing_quotes import healing

from money_quotes import money

trained_model, vocabulary = AI.making_the_model()

while True:
    
    user_input = input("Enter a sentence to get an Appropriate Quote (q to Quit): ").lower()
    
    vectorized_input = vocabulary.transform([user_input])

    prediction = trained_model.predict(vectorized_input)

    if user_input == "quit":
        print("Thanks for chating with me... ")
        break

    random_index = random.randint(1, 50)

    def check_money():
        if prediction == "money":
            money_quotes = money()
            print(money_quotes[random_index])

    check_money()


    def check_motivation():
        if prediction == "motivation":
            motivation_quotes = motivation()
            print(motivation_quotes[random_index])

    check_motivation()

    def check_god():
        if prediction == "god":
            god_quotes = god()
            print(god_quotes[random_index])

    check_god()

    def healing_check():
        if prediction == "healing":
            healing_quotes = healing()
            print(healing_quotes[random_index])

    healing_check()