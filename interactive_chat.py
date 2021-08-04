import sys

import naive_bayes_bot

import decision_tree_bot

chat_bot_usage = "To run the chat bot please input program name and desired bot type: " \
                 "--decisiontree --naivebayes --neuralnet or --markovchain"

def check_arguments():
    n = len(sys.argv)
    if n < 2:
        print(chat_bot_usage)
        exit()
    if sys.argv[1] == "--decisiontree":
        return decision_tree_bot.DecisionTreeBot()
    if sys.argv[1] == "--naivebayes":
        return naive_bayes_bot.NaiveBayesBot()
    if sys.argv[1] == "--neuralnet":
        return #neural net bot
    if sys.argv[1] == "--markovchain":
        return #markov chain bot
    else:
        print(chat_bot_usage)
        exit()

def run_chat():
    bot = check_arguments()
    while True:
        user_message = input("Type your message here:")
        if user_message == "quit" or user_message == "exit":
            break
        print(bot.formulate_response(user_message))


if __name__ == '__main__':
    run_chat()

