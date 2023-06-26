import csv
from n_gram import NGram

def read_csv(filepath):
    with open(filepath) as infile:
        l = []
        rdr = csv.reader(infile, delimiter=',')
        for i in rdr:
            a = [ ]
            for j in i:
                a.append(j)
            l.append(i)
        return l
    
print('Formatting data')
twitter_training = read_csv('data/twitter_training.csv')
twitter_validation = read_csv('data/twitter_validation.csv')

def get_data(data):
    positive_sentences = []
    negative_sentences = []
    neutral_sentences = []
    irrelevant_sentences = []
    for row in data:
        match row[2]:
            case "Positive":
                positive_sentences.append(row[3])
            case "Negative":
                negative_sentences.append(row[3])
            case "Neutral":
                neutral_sentences.append(row[3])
            case "Irrelevant":
                irrelevant_sentences.append(row[3])

    return [
        ("Positive", positive_sentences),
        ("Negative", negative_sentences),
        ("Neutral", neutral_sentences),
        ("Irrelevant", irrelevant_sentences)
    ]

n_gram_input_training = get_data(twitter_training)
n_gram_input_validation = get_data(twitter_validation)

print('Data formatted')
n_gram = NGram(n_gram_input_training, 3)

print('Trained')

print(n_gram.validate(n_gram_input_validation))