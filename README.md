# Domain Agnostic Real-Valued Specificity Prediction
Code for
Wei-Jen Ko, Greg Durrett and Junyi Jessy Li, "Domain Agnostic Real-Valued Specificity Prediction", The AAAI Conference on Artificial Intelligence (AAAI), 2019

This is a text specificity predictor for any domain. 

**Citation:**
```
@InProceedings{ko2019domain,
  author    = {Ko, Wei-Jen and Durrett, Greg and Li, Junyi Jessy},
  title     = {Domain Agnostic Real-Valued Specificity Prediction},
  booktitle = {AAAI},
  year      = {2019},
}
```


## Dependencies
-Pytorch (Tested on 1.0.0, it is known to produce incorrect results on 1.7.1)

-Numpy

## Data and resources
The glove vector file (840B.300d) is required. Download it and set the glove path in train.py and test.py
```
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```
The twitter, yelp, and movies data and annotations used in the paper is in dataset/data

Data format:

twitters.txt contains the sentences that have corresponding labels in twitterv.txt and twitterl.txt ("s" for sentences).

twitterv.txt are average specificity labeled by turkers for each sentence in twitters.txt, permutated in the same order ("v" for values).

twitteru.txt is the unlabeled target domain data used in the paper ("u" for unlabeled).

twitterl.txt are the binary specificity labels, which are not used ("l" for labeled).

For other domains, substitute the domain name in the file names.

## Running 
Training command:

python train.py  --gpu_id 0 --test_data twitter

Testing command:

python test.py  --gpu_id 0 --test_data twitter

## Using it on a new domain
To use it on a new domain, unlabeled sentences of the new domain is required.

When training,change the s1['unlab']['path'] in data2.py and the path of xsu in train.py and test.py to the unlabeled data file.

When testing, change the s1['test']['path'] in data2.py and the path of xst in test.py to the test sentences file.(And make sure s1['unlab']['path'] in data2.py and the path of xsu in test.py is the same file)

The unlabeled data can be the same as test data.

<b>The first line in the testing data is ignored.</b>

### Steps from beginning to end (assumes GPU)

- git clone https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction.git
- cd Domain-Agnostic-Sentence-Specificity-Prediction
- wget https://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip
- unzip glove.840B.300d.zip
- python train.py --gpu_id 0 --test_data twitter
- python test.py  --gpu_id 0 --test_data twitter

## For users without GPU/CUDA
Please replace the train.py, test.py, model.py with the files in no_cuda.zip
