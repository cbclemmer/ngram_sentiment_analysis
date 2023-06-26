from itertools import groupby
from operator import itemgetter

from typing import Dict, List, Tuple

def create_grams(s: str, n: int) -> List[str]:
    ret_val = [ ]
    last_words = [ ]
    for wd in s.split(' '):
        if wd == '':
            continue
        last_words.append(wd)
        if len(last_words) < n:
            continue
        gram = ''
        first = True
        for g_wd in last_words:
            w = ''
            if first:
                first = False
                w = g_wd
            else:
                w = ' ' + g_wd
            gram += w
        last_words.pop(0)
        ret_val.append(gram)
    return ret_val

def flatten(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

class NGram:
    def __init__(self, input_data: List[Tuple[str, List[str]]], max_grams: int):
        self.ngram_maps = [ ]
        self.max_grams = max_grams
        self.train(input_data)

    def train(self, input_data: List[Tuple[str, List[str]]]) -> None:
        total_words = 0
        for (_, sentences) in input_data:
            for sentence in sentences:
                total_words += len(sentence.split(' '))

        for num_grams in range(1, self.max_grams + 1):
            bm = {}
            for sentiment, wv_input in input_data:
                bm[sentiment] = self.train_gram_vector(wv_input, num_grams, total_words)
            self.ngram_maps.append(bm)

    def train_gram_vector(self, input_data: List[str], num_grams: int, total_words: int) -> Dict[str, Dict[str, float]]:
        grams = []
        for s in input_data:
            for g in create_grams(s, num_grams):
                grams.append(g)

        total_grams = total_words / num_grams
        total_wv = { }
        for gram in grams:
            if gram in total_wv:
                total_wv[gram] += 1
            else:
                total_wv[gram] = 1
        
        prob_wv = { }
        for gram in total_wv.keys():
            prob_wv[gram] = total_wv[gram] / total_grams

        return prob_wv

    def test_gram(self, gram: str) -> str:
        best_prob = ("", 0.0)
        gram_length = len(gram.split(' '))
        for sentiment in self.ngram_maps[gram_length - 1].keys():
            g_map = self.ngram_maps[gram_length - 1][sentiment]
            prob = g_map.get(gram)
            if prob is None:
                continue
            if prob > best_prob[1]:
                best_prob = (sentiment, prob)
        return best_prob[0]

    def test_sentence(self, sentence: str) -> str:
        totals_hm = {}
        found_words = []
        for i in range(len(self.ngram_maps), 0, -1):
            for gram in create_grams(sentence, i):
                best_type = self.test_gram(gram)
                if best_type == "":
                    continue
                gram_words = gram.split(" ")
                first_word = gram_words[0]
                if first_word in found_words:
                    continue
                else:
                    found_words.append(first_word)
                total = totals_hm.get(best_type, 0) + 1
                totals_hm[best_type] = total

        best_type = ("", 0)
        for type_name, total in totals_hm.items():
            if total > best_type[1]:
                best_type = (type_name, total)

        if best_type[0] == "":
            best_type = ("Inconclusive", best_type[1])

        return best_type[0]

    def validate(self, input_data: List[Tuple[str, List[str]]]) -> float:
        num_sentences = 0
        num_correct = 0

        for sentiment, sentences in input_data:
            for sentence in sentences:
                num_sentences += 1
                predicted_type = self.test_sentence(sentence)
                if predicted_type == sentiment:
                    num_correct += 1

        return num_correct / num_sentences