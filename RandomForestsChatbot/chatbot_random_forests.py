import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

with open("question.json", 'r') as file:
    questions = json.load(file)
    file.close()

with open("answer.json") as file:
    answers = json.load(file)
    file.close()

# Prepare the training data
training = {}

for intent, questions in questions.items():
    for ques in questions:
        training[ques] = intent

# Seperate out the intents and questions
responses = np.array(list(training.keys()))
class_labels = np.array(list(training.values()))

# Convert to vector
vector = TfidfVectorizer().fit(responses)
x = vector.transform(responses).toarray()

y = class_labels.reshape(-1, 1)

# Create the classifier and model
chatbot_model = RandomForestClassifier(n_estimators=200)
chatbot_model.fit(x, y)

# create the interface
def interface(question):
    process = vector.transform([question]).toarray()
    probability = chatbot_model.predict_proba(process)[0]
    max_out  = np.argmax(probability)
    if probability[max_out] < 0.55:
        return "Sorry, I don't understand..."
    else:
        return answers[chatbot_model.classes_[max_out]]
while True:
    user = input("User>> ")
    if user == "quit":
        break
    print("SmartBot>> {}".format(interface(user)))