## Author: Jessy Li (ljunyi@seas.upenn.edu)

## given raw text files, generate features for
## shallow and neuralbrn classifiers

from collections import namedtuple
import os.path

import features
import utils
import numpy as np
from data2 import get_batch
import torch
from torch.autograd import Variable

Instance = namedtuple("Instance","uid,label,rawsent")
word_vectors = {}

class ModelNewText(object):

    def __init__(self, brnspace=None, brnclst=None, embeddings=None, tokenizer="split"):
        self.featurestest = {} ## <name, flist>
        self.test = []
        self.brnclst = brnclst
        self.brnspace = brnspace
        self.embeddings = embeddings
        self.fileid = None
        self.tokenizer = tokenizer

    def loadFromFile(self,filename):
        self.test = []
        self.fileid = os.path.basename(filename)
        i = 0
        with open(filename) as f:
            for line in f:
                if len(line.strip()) == 0: 
                    print(i)
                    #continue
                self.test.append(Instance(self.fileid+"."+str(i),0,features.RawSent(line.strip(), tokenizer=self.tokenizer)))
                i += 1
        f.close()

    def loadSentences(self, identifier, sentlist):
        ## sentlist should be a list of sentence strings, tokenized;
        ## identifier is a string serving as the header of this sentlst
        self.test = []
        self.fileid = identifier
        for i,sent in enumerate(sentlist):
            # sent="".join(sent[1:-1])
            self.test.append(Instance(identifier+"."+str(i),0,features.RawSent(sent, tokenizer=self.tokenizer)))
            
    def _add_feature(self, key, values):
        if key in self.featurestest: return
        self.featurestest[key] = values
    
    def fShallow(self):
        normalize = True
        recs = [r.rawsent for r in self.test]
        self._add_feature("sentlen",features.sentLen(recs))
        self._add_feature("numnumbers",features.numNumbers(recs, normalize))
        self._add_feature("numcapltrs",features.numCapLetters(recs, normalize))
        self._add_feature("numsymbols",features.numSymbols(recs, normalize))
        self._add_feature("avgwordlen",features.avgWordLen(recs))
        self._add_feature("numconns",features.numConnectives(recs))
        self._add_feature("fracstopwords",features.fracStopwords(recs))
        polarvals = features.mpqaGenInqInfo(recs)
        keys = ["mpqageninq-subj","mpqageninq-polarity"]
        for (key,vals) in zip(keys,polarvals):
            self._add_feature(key,vals)
        mrcvals = features.mrcInfo(recs)
        keys = ["mrc-fami","mrc-img"]
        for (key,vals) in zip(keys,mrcvals):
            self._add_feature(key,vals)
        idfvals = features.idf(recs)
        keys = ["idf-min", "idf-max", "idf-avg"]
        for (key,vals) in zip(keys,idfvals):
            self._add_feature(key,vals)
        
    def fNeuralVec(self):
        keys = ["neuralvec-"+str(i) for i in range(100)]
        if keys[0] not in self.featurestest:
            feats = features.neuralvec(self.embeddings,[r.rawsent for r in self.test])
            for i,key in enumerate(keys):
                self.featurestest[key] = feats[i]

    def fBrownCluster(self):
        if self.brnclst == None:
            self.brnclst = utils.readMetaOptimizeBrownCluster()
        key = "brnclst1gram"
        if key not in self.featurestest:
            self.featurestest[key] = []
            for instance in self.test:
                rs = features.getBrownClusNgram(instance.rawsent,1,self.brnclst)
                rs = ["_".join(x) for x in rs]
                self.featurestest[key].append(rs)

    def transformShallow(self):
        ys = [x.label for x in self.test]
#        xs = [{} for i in range(len(self.test))]
        xs = np.zeros((len(self.test),14))

        fnames = ["sentlen","numnumbers","numcapltrs","numsymbols","avgwordlen","numconns","fracstopwords","mpqageninq-subj","mpqageninq-polarity","mrc-fami","mrc-img","idf-min","idf-max","idf-avg"]
        for fid,fname in enumerate(fnames):
            for i,item in enumerate(self.featurestest[fname]):
                xs[i,fid] = item
        return ys,xs

    def transformWordRep(self):
        neuralvec_start = 1
        ys = [x.label for x in self.test]
        xs = [{} for i in range(len(self.test))]
        for j in range(100):
            fname = "neuralvec-"+str(j)
            for i,item in enumerate(self.featurestest[fname]):
                xs[i][j+1] = item
        for i,item in enumerate(self.featurestest["brnclst1gram"]):
            xs[i].update(self.brnspace.toFeatDict(item,False))
        return ys,xs

    def __call__(self, sentences: list, word_vectors_path: str='glove.840B.300d.txt', use_gpu: str=True):
        """
        Given sentences to score for specificity, 
        return what is expected as input features to the model.

        Args:
            - sentences (list of str)
            - word_vectors_path (str): For example, glove.840B.300d.txt
            - use_gpu (bool): If True, will use GPU/cuda.

        Returns:
            - sentence_embeddings
            - sentence_lens
            - feature_variable

            The above are returned as follows:
            - (sentence_embeddings, sentence_lens), feature_variables
            Because this is what the model expects as input.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        # Ensure there's no carry over of data from previous calls.
        self.test = []
        self.featurestest = {}
        # Prepare and store in memory features for the sentences loaded.
        self.loadSentences("new", sentences)
        self.fShallow()
        # Turn the features into a numpy vector.
        labels, features = self.transformShallow()

        # Turn the features into a numpy vector.
        global word_vectors

        # Load word vectors.
        if not word_vectors:
            with open(word_vectors_path, encoding="utf8") as file:
                print('ModelNewText: Loading word vectors from', word_vectors_path)
                for line in file:
                    word, vec = line.split(' ', 1)
                    if word:
                        try:
                            word_vectors[word] = np.array(list(map(float, vec.split())))
                        except Exception:
                            import traceback
                            print(traceback.format_exc())
                            print('Error loading word vector for word', word)
                print(f'ModelNewText: Loaded {len(word_vectors)} word vectors.')
        sentence_tokens = [sent.rawsent.getTokens() for sent in self.test]
        # Prepare a torch vector of the GloVe word vectors 
        # for the sentence and a vector of sentence lengths.
        sentence_embeddings, sentence_lens = get_batch(sentence_tokens, word_vectors, 300)

        # Prepare Torch Variable for the features.
        feature_vector = torch.from_numpy(features).float()
        feature_variable = Variable(feature_vector)
        if use_gpu:
            feature_variable = feature_variable.cuda()
            sentence_embeddings = sentence_embeddings.cuda()
        
        return (sentence_embeddings, sentence_lens), feature_variable
