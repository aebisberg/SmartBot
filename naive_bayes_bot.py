#A class for implementing a bot that used multinomial naive bayes to answer simple chat questions
#using training data and answers given to the bot
#
#Use formulate_response(input_text) to recieve a response to a query

import re
import string
from collections import defaultdict

class NaiveBayesBot:

    def __init__(self):
        #Set up data for training
        queries, stop_words = {}, set()
        for line in open('stop_words.txt'):
            words = line.split()
            if not re.match('^#',words[0]):
                stop_words.add(words[0])
        for line in open('naive_training_data.txt'):
            contents = line.strip().split(" ")
            label = contents.pop(len(contents)-1)
            helpful_words = list(set(contents) - stop_words)  #remove common stop words from the list
            if label not in queries.keys():
                queries[label] = [helpful_words]
            else:
                queries[label].append(helpful_words)
        answers = []
        for line in open('naive_responses.txt'):
            answers.append(line)
        output = self.__prepare_data__(queries)

        self.trained_words = output[0]
        self.total_counts = output[1]
        self.class_counts = output[2]
        self.prior_probs = output[3]
        self.answers = answers

    #get relevent counts for words across classes, total appearence of words, and total words in classes
    def __prepare_data__(self, queries):
        total_words, class_counts,  total_counts = 0, defaultdict(), defaultdict()
        for key in queries.keys():
            total_class, counts = 0, {}
            for sentence in queries[key]:
                for word in sentence:
                    total_class += 1
                    total_words += 1
                    if word not in counts.keys():
                        counts[word] = 1
                    if word not in total_counts.keys():
                        total_counts[word] = 1
                    else:
                        counts[word] += 1
                        total_counts[word] += 1
            class_counts[key] = total_class, counts
        prior_class_probs = {}
        for key in class_counts.keys():
            prior_class_probs[key] = class_counts[key][0] / total_words
        return total_words,total_counts,class_counts, prior_class_probs

    #calculates the bayes distribution of a given word
    def __bayes_distribution__(self, given_word):
        count_dict, p_of_class = self.class_counts, defaultdict()
        for class_name in count_dict.keys():
            p_class_given_word, p_class = 0, self.prior_probs[class_name]
            if given_word in count_dict[class_name][1].keys():
                p_word_given_class = count_dict[class_name][1].get(given_word) / count_dict[class_name][0]
                p_word = self.total_counts.get(given_word)
                p_class_given_word = (p_word_given_class * p_class) / p_word
            p_of_class[class_name] = p_class_given_word
        return p_of_class

    def formulate_response(self, input_text):
        input_text.translate(str.maketrans('', '', string.punctuation))
        split_text = list(input_text.strip().split(" "))
        cumulative = defaultdict()

        for word in split_text:
            #get the bayes distribution of each word
            answer_probs = self.__bayes_distribution__(word)
            #accumulate the probabilities for each word
            for key in answer_probs:
                if key not in cumulative.keys():
                    cumulative[key] = answer_probs[key]
                else:
                    if cumulative[key] == 0.0:
                        cumulative[key] = answer_probs[key]
                    else:
                        cumulative[key] = cumulative[key] * answer_probs[key]
        best = 0

        for key in cumulative: #searching for the maximum key
            if cumulative[key] > best:
                best = int(key)
        if best == 0:
            return self.answers.__getitem__(len(self.answers)-1) #no good match to ask again
        return self.answers.__getitem__(best-1) #answer with the best match



