import string

class Sms:
    def set_text(self, text):
        self.text = text
    
    def __init__(self,line, sep = ",", only_text = False):
        if only_text:
            self.text = line
            self.label = ""
        else:
            line = line.strip().split(sep)
            self.label = line[0]
            translator = str.maketrans('', '', string.punctuation)
            self.text = line[1].translate(translator).strip()

    def get_all_words(self):
        words = self.text.split(" ")
        words_stripped = []
        for word in words:
            words_stripped.append(word.strip())
        return words_stripped

    def get_set_of_words(self):
        words = self.text.split(" ")
        words_stripped = set()
        for word in words:
            words_stripped.add(word.strip())
        return words_stripped

from enum import Enum, auto

class Evaluation(Enum):
    K_FOLD_CROSS = auto()
    RANDOM_SPLIT = auto()   #not supported yet

class Distribution(Enum):
    BERNOULLI = auto()
    MULTINOMIAL = auto()

class NaiveBayesText:
    def __init__(self, labels = ["yes", "no"]):
        self.labels = labels
        self.evaluation = Evaluation.K_FOLD_CROSS
        self.distribution = Distribution.BERNOULLI
        self.k_fold = 10
        #self.seed = 42  #don't know why 42 :)
    
    def set_k_fold(self, k):
        self.k_fold = k

    def set_distribution(self, distribution):
        if distribution in Distribution:
            self.distribution = distribution

    #def set_seed(self, seed):
    #    self.seed = seed

    def _dataset_to_csv(filename, header=True):
        csv_filename = filename + ".csv"
        fin = open(filename, "r")
        fout = open(csv_filename, "w")
        if header:
            print("label,text", file=fout)

        for line in fin:
            line = line.strip()
            index = line.find('\t')
            label = line[:index]
            text = line[index:].strip().replace(",", ";")
            print(label, text, sep = ",", file = fout)
        fin.close()
        fout.close()
        return csv_filename

    def _csv_to_docs(self, csv_filename):
        self.docs = {}
        for l in self.labels:
            self.docs[l] = []

        self.words = set()
        fin = open(csv_filename, "r")
        fin.readline() #header
        for line in fin:
            s = Sms(line)
            self.docs[s.label].append(s.text)
            self.words = self.words.union(s.get_set_of_words())
        fin.close()

    def _set_training_and_testing_data(self, test_id = 0):
        self.training_docs = {}
        self.testing_docs = {}

        if self.evaluation == Evaluation.K_FOLD_CROSS:
            for label in self.labels:
                self.training_docs[label] = []
                self.testing_docs[label] = [] 

                l = self.docs[label].copy()
                while(l):
                    for i in range(self.k_fold):
                        if not l:
                            break
                        elif i == test_id:
                            self.testing_docs[label].append(l.pop())
                        else:
                            self.training_docs[label].append(l.pop())

    def _train(self):
        self.t = {}
        for label in self.labels:
            self.t[label] = len(self.training_docs[label])

        self.tij = {}
        for w in self.words:
            self.tij[w] = {}
            for label in self.labels:
                freq = 0
                for text in self.training_docs[label]:
                    if self.distribution == Distribution.BERNOULLI:
                        if w in text:
                            freq+=1
                    elif self.distribution == Distribution.MULTINOMIAL:
                        freq+= text.count(w)
                self.tij[w][label] = freq

    def _classify(self, new_text):
        v = len(self.words)
        if self.distribution == Distribution.BERNOULLI:
            laplace_par = 2
        elif self.distribution == Distribution.MULTINOMIAL:
            laplace_par = v
        
        new_text_words = Sms(new_text, only_text = True).get_all_words()
        p_labels = {}
        for label in self.labels:
            p_labels[label] = 1
        for w in new_text_words:
            w = w.strip()
            if w in self.words:
                for label in self.labels:
                    p_labels[label]*= (self.tij[w][label] + 1)/(self.t[label] + laplace_par)
        max_p = -1 # handle a single case where len(msg) > 100 and p goes to 0 
        max_label = ""
        for label in self.labels:
            prob = p_labels[label]*self.t[label]/v
            if prob > max_p:
                max_p = prob
                max_label = label
        return max_label

    def _create_confusion_matrix(self):
        self.confusion_matrix = {}
        for lab_1 in self.labels:
            self.confusion_matrix[lab_1] = {}
            for lab_2 in self.labels:
                self.confusion_matrix[lab_1][lab_2] = 0

        for label in self.labels:
            l = self.testing_docs[label]
            for text in l:
                h = self._classify(text)
                if h not in self.labels:
                    print(h, text)
                else:
                    self.confusion_matrix[label][h]+=1

                
    def set_evaluation_method(self, evaluation):
        if evaluation in Evaluation:
            self.evaluation = evaluation
    
    def set_dataset(self, filename, csv=False):
        if not csv:
            filename = NaiveBayesText.dataset_to_csv(filename)
        self._csv_to_docs(filename)

    def _learn(self, i):
        self._set_training_and_testing_data(test_id = i)
        self._train()

    def evaluate(self):
        if self.evaluation == Evaluation.K_FOLD_CROSS:
            delta = 0
            for i in range(self.k_fold):
                self._learn(i)
                self. _create_confusion_matrix()
                tot_err = 0
                tot_ok = 0
                for lab_1 in self.labels:
                    for  lab_2 in self.labels:
                        if lab_1 == lab_2:
                            tot_ok += self.confusion_matrix[lab_1][lab_2]
                        else:
                            tot_err+= self.confusion_matrix[lab_1][lab_2]

                delta+= tot_err/(tot_ok + tot_err)
                print(delta/(i+1))
                

    def classify_single(self, text):
       return self._classify(text)

def main():
    m = NaiveBayesText(labels=["ham", "spam"])
    m.set_dataset("SMSSpamCollection.csv", csv = True)
    m.set_distribution(Distribution.MULTINOMIAL)
    m.evaluate()
    return m

m = main()
