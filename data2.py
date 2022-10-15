
import os
import numpy as np
import torch


def get_batch(batch, word_vec,wed):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), wed))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            try:
                embed[j, i, :] = word_vec[batch[i][j]]
            except KeyError:
                pass

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}

    dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test
    
def get_pdtb(data_path,dom,unsupervised_data_name,tv,supervised_data_name=None):
    """
    Args:
        - supervised_data_name (str): This is used for the supervised portion of the training. If "news", it will use the files data_path/news_sentences.txt and data_path/news_labels.txt.
        - unsupervised_data_name (str): One of 'twitter', 'yelp', 'movie'
            or another you define and add files for.
            Will be used to identify data files.
    """
    s1 = {}
    s2 = {}
    target = {}
    
    targetv = {}
    
    dico_label = {'1': 0,  '2': 1}
    
    for data_type in ['trainu','train','unlab','test']:
        s1[data_type], s2[data_type], target[data_type],targetv[data_type] = {},{}, {}, {}
    if supervised_data_name:
        s1['train']['path'] = os.path.join(data_path, f'{supervised_data_name}_sentences.txt')
    

    s1['test']['path'] = os.path.join(data_path, f'{unsupervised_data_name}_sentences.txt') # Sentences
    s1['unlab']['path'] = os.path.join(data_path, f'{unsupervised_data_name}_unlabeled_sentences.txt') # Unlabeled sentences
    # Note: The data specified by the below target and targetv are not used except as dummy variables,
    # but if not provided or not existing there will be an error.
    target['test']['path'] = 'dataset/data/twitter_labels.txt' # Specificity binary labels for sentences
    targetv['test']['path'] = 'dataset/data/twitter_ratings.txt' # Spelabelcificity ratings for sentences

    s1['trainu']['path'] = os.path.join(data_path, 'aaai15unlabeled/all.60000.sents')

    if supervised_data_name:
        target['train']['path'] = os.path.join(data_path,f'{supervised_data_name}_labels.txt')

    target['trainu']['path'] = os.path.join(data_path,'aaai15unlabeled/all.60000.spec')

    if supervised_data_name:
        s1['train']['sent'] = [line.rstrip() for line in open(s1['train']['path'], 'r')]
    s1['unlab']['sent'] = [line.rstrip() for line in open(s1['unlab']['path'], 'r')]
    s1['test']['sent'] = [line.rstrip() for line in open(s1['test']['path'], 'r')]
    s1['trainu']['sent'] = [line.rstrip() for line in open(s1['trainu']['path'], 'r')]
   
    if supervised_data_name:
        target['train']['data'] = np.array([dico_label[line.rstrip('\n')]
                    for line in open(target['train']['path'], 'r')])
    
    try:
        target['test']['data'] = np.array([dico_label[line.rstrip('\n')]
                    for line in open(target['test']['path'], 'r')])
    except KeyError:
        raise KeyError(f'Invalid label found, where the options are {dico_label}\nFilepath with invalid label: {target["test"]["path"]}')

    targetv['test']['data'] = np.array([float(line.rstrip('\n'))
                for line in open(targetv['test']['path'], 'r')])
    target['trainu']['data'] = np.array([int(float(line.rstrip('\n'))>0.5)
                for line in open(target['trainu']['path'], 'r')])
    if not (unsupervised_data_name=='subso'):   
        assert len(s1['train']['sent'])== len(target['train']['data'])

    print('** {0} DATA : Found {1} of {2} sentences.'.format(data_type.upper(), len(s1['train']['sent']), 'train'))
    if unsupervised_data_name=='twi':   
        train = {'s1': s1['test']['sent'][:tv],# 's2': s2['train']['sent'],
                 'label': target['test']['data'][:tv]}
    elif unsupervised_data_name=='pdtb':
        train = {'s1': s1['train']['sent'][:2784],# 's2': s2['train']['sent'],
                 'label': target['train']['data'][:2784]}
    elif unsupervised_data_name=='pdtb2':
        train = {'s1': s1['train']['sent'][:49280],# 's2': s2['train']['sent'],
                 'label': target['train']['data'][:49280]}
    
    elif dom==1:   
        train = {'s1': s1['train']['sent'],# 's2': s2['train']['sent'],
                 'label': target['train']['data']}
    
    elif dom==2:   
        train = {'s1': s1['train']['sent'][:2000],# 's2': s2['train']['sent'],
                 'label': target['train']['data'][:2000]}
        
    else:
        train = {'s1': s1['train']['sent'][:2877],# 's2': s2['train']['sent'],
                 'label': target['train']['data'][:2877]}

    unlab = {'s1': s1['unlab']['sent']}#, 's2': s2['train']['sent'],
         #    'label': target['train']['data']}
    trainu = {'s1': s1['trainu']['sent'],# 's2': s2['train']['sent'],
             'label': target['trainu']['data']}
    dev = {'s1': s1['test']['sent'][:tv],'label': target['test']['data'][:tv],'labelv': targetv['test']['data'][:tv]}
    test = {'s1': s1['test']['sent'][tv:],'label': target['test']['data'][tv:],'labelv': targetv['test']['data'][tv:]}
 
    return train, dev, test,unlab,trainu
