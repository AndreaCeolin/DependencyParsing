### **CIS 530 - Final project**
# Dependency Parsing
Andrea Ceolin, Chenxi Li, Daizhen Li, Mingyang Liu, Linzhi Qi, Veronica Qing Lyu

## Abstract

Finding the dependency path between words is a complex task, but the NLP community has always been interested in this topic. This project explores different methods of implementing a dependency parser for English. An intuitive left-arc algorithm is exploited to build a simple baseline. Two published baselines are then implemented, either from the graph-based or the transition-based paradigm. On top of them, a set of extensions with the introduction of neural networks are tried out for improving the baseline performance. 

## 1. Introduction

Dependency grammar is an important class of grammar formalism in contemporary speech and language processing systems. Dependency is the notion that linguistic units, e.g. words, are connected to each other by directed links. The purpose of our project is to build a model for dependency parsing of English, and output the dependency path in the format specified by the CoNLL shared task 2017. Conventional evaluation metrics include Labeled Attachment Score(LAS),  Unlabeled Attachment Score(UAS) and so on. 

The structural center of the dependence path is a verb, and all other syntactic units are directly or indirectly related to this verb, as shown in the figure below. The pronoun "I" and the noun "Python" are both linked to the verb "love" with their corresponding dependency path. 
![Figure1 - Parsed sentence for English. In a simple case like this one, both the subject and the object of a sentence depend on the verb](https://i.imgur.com/Yv9TfHq.png)
_Figure1 - Parsed sentence for English. In a simple case like this one, both the subject and the object of a sentence depend on their verb._

The parser will take in data as .conllu form, with sentences and their tokens including Id; Form; Lemma; UPosTag; XPosTag;Feats; Head; DepRel; Deps; Misc. The parser will process the sentences and determine the dependency path between each pair and the linguistic units and label them. 
 

The reason why we picked this topic to be our final project is that the multilingual dependency parser in CoNLL shared task 2018 interested us the most. As the project went along, we found that the multilingual parser requires a significant amount of features and the training time is also very long. Therefore, we trimmed the project down to just a dependency parser for English and continuously worked to improve the performance of our models. 



## 2. Literature review

There are many different approaches to Dependency Parsing, and they are radically different when it comes to the model architecture.

For instance, one could try a simple unsupervised rule-based approach by hard-coding some rules connecting Part of Speech (POS) together. This was the case of the algorithm in Garcia and Gamallo (2017), which implemented a rule-based parser which encoded syntactic rules found in Romance languages, called MetaRomance. MetaRomance is a delexicalized model that runs on Universal POS tags. Since it's a rule-based model, it requires no training data. 150 linguistic rules were encoded in the algorithm. The parser achieved good results for Romance languages (UAS: 71% for Italian, 69% for Portuguese and Spanish, 65 for Catalan, 63% for Galician and French), but its performance went drastically down when tested for non-Indoeuropean languages (UAS: 51% for Indonesian, 46% for Hungarian, 38% for Hebrew, 37% for Estonian and Arabic were the best results). Interestingly, for Japanese the UAS performance went as down as 8%.

A different approach has been proposed in McDonald and Pereira (2005). This paper presents an algorithm for parsing dependency trees using as a model maximum spanning trees in directed graphs. First, the authors factor the score of a dependency tree as the sum of the scores of all edges in the tree. The score of each edge is the dot product between feature representation of the edge and a weight vector parameter. Then, the parser uses online large-margin learning to learn parameter by narrowing the score of true denpendency tree with current highest score. 

This method can achieve 84% in Accuracy and 32% in completeness in Czech. For non-projective Czech sentences, their method get 6% improvement from projective method and also get 14% proportion exactly correct. For English, they can achieve state-of-art performance: 90% in Accuracy and 33% in completeness. Their Accuracy metric is equivalent to LAS metric in Conll2017.

More recently, a variety of more advanced neural network architectures have been used to address the task. Dozat et al. (2016, 2017) describe a graph-based neural dependency parser implemented by Stanford for the CoNLL 2017 shared task. Their parser employs the so-called "deep-biaffine" mechanism to produce POS tags and labeled dependency parses from segmented and tokenized sequences of words. It also includes a character-based representation that uses an LSTM to produce embeddings from sequences of characters, in order to address the issue of rare word. The parser was ranked first among all systems submitted to the shared task, with an averaged LAS score of 76% for 49 languages.

A similar approach has been developed in Shi et. al. (2017), which presents a character-level bi-directional LSTM as lexical feature extractor and combining graph-based and transition-based global parsing paradigms. The system relies on the baseline tokenizers and focuses only on parsing, leveraging bi-LSTMs to generate compact features for both graph-based and transition-based parsing frameworks. One graph-based paradigm and two transition-based paradigms are used. The system performs four procedures, UDPipe pre-processing, feature extraction by character bi-LSTM, unlabeled parsing with global parsing paradigms and arc labeling. The parser achieved LAS F1 score of 75%, with 47% for surprise languages and 61% for small treebanks. The system ranked second among all systems submitted to the shared task, and first for both surprise languages and small treebanks. 

## 3. Experimental design
### 3.1 Data

Our training, development and testing data are retrieved from the database of the [CoNLL 2017 Shared Task](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2184). For this project, we worked with the English treebank.

The sentences have been manually annotated using the *.conllu* format. In this format, the first row contains the sentence to be parsed. Then in the following rows, the first column contains the index of the word, the second column denotes the word, the fifth column shows the Part-of-Speech of the word and the seventh column contains the head of the word. For the purposes of our task, we can ignore all the other columns, which usually provide additional information about the arcs, or a more fine-grained description of the words and the POS which is language specific.


	#text = Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.
	1	Al		_	_	NNP	_	0	_	_	_
	2	-		_	_	HYPH	_	1	_	_	_
	3	Zaman		_	_	NNP	_	1	_	_	_
	4	:		_	_	:	_	1	_	_	_
	5	American	_	_	JJ	_	6	_	_	_
	6	forces		_	_	NNS	_	7	_	_	_
	7	killed		_	_	VBD	_	1	_	_	_
	8	Shaikh		_	_	NNP	_	7	_	_	_
	9	Abdullah	_	_	NNP	_	8	_	_	_
	10	al		_	_	NNP	_	8	_	_	_
	11	-		_	_	HYPH	_	8	_	_	_
	12	Ani		_	_	NNP	_	8	_	_	_
	13	,		_	_	,	_	8	_	_	_
	14	the		_	_	DT	_	15	_	_	_
	15	preacher	_	_	NN	_	8	_	_	_
	16	at		_	_	IN	_	18	_	_	_
	17	the		_	_	DT	_	18	_	_	_
	18	mosque		_	_	NN	_	7	_	_	_
	19	in		_	_	IN	_	21	_	_	_
	20	the		_	_	DT	_	21	_	_	_
	21	town		_	_	NN	_	18	_	_	_
	22	of		_	_	IN	_	23	_	_	_
	23	Qaim		_	_	NNP	_	21	_	_	_
	24	,		_	_	,	_	21	_	_	_
	25	near		_	_	IN	_	28	_	_	_
	26	the		_	_	DT	_	28	_	_	_	
	27	Syrian		_	_	JJ	_	28	_	_	_
	28	border		_	_	NN	_	21	_	_	_
	29	.		_	_	.	_	1	_	_	_
	
In the test file, the seventh column is missing, and we are required to fill it with a number that refers to its heads.
The dependency tree can be visualized on [displaCy](https://explosion.ai/demos/displacy?text=Al-Zaman%20%3A%20American%20forces%20killed%20Shaikh%20Abdullah%20al-Ani%2C%20the%20preacher%20at%20the%20mosque%20in%20the%20town%20of%20Qaim%2C%20near%20the%20Syrian%20border.%20&model=en_core_web_sm&cpu=1&cph=0).


### 3.2 Evaluation Metric

A simple evaluation script was provided by the CoNLL 2017 organizers to the participant of the shared task [here](http://universaldependencies.org/conll17/evaluation.html). This script takes two *.conllu* files, a gold file and a predicted output file, and calculates the F1-score (which in dependency parsing is defines as **UAS, Unlabel Attachment Score**):

    py score.py goldconllufile predconllufile
    
UAS is a standard evaluation metric in parsing: the percentage of words that are assigned the correct syntactic head. Precision, P, is defined as the number of correct relations divided by the number of system-produced nodes. Recall, R, is defined as the number of correct relations divided by the number of gold-standard nodes. Finally, UAS is calculated as the F1 score=2PR/(P+R). The goal of our task is maximizing the UAS score.



### 3.3 Simple Baseline
Languages tend to have a strong preference for 'harmonic' word order: for instance, languages in which articles precedes nouns usually also have prepositions preceding nouns. For this reason, one needs to determine how many dependencies can be detected just by orienting the arcs in the direction that is favored by the language. English is a language that usually has the head of the constituent on the right: for instance, elements that are related to nouns as determiners, adjective or other nouns in compounds always come first, and similarly subject and adverbs tend to precede the verb they relate to. 

![Parsed sentence for English. Most of the arcs have their head on the right](https://image.ibb.co/me4rMx/English.png)
_Figure2 - Parsed sentence for English. Most of the words have their head on the right_

With this purpose, we built a simple baseline for which every word has as a head the word on its right. This strategy allowed us to obtain a 30% UAS on the English dataset. Since these 'harmony' tendencies are more or less universal, one can assume that any parser for any language should yield a performance above this number to be more informative than a simple majority baseline.

## 4. Experimental results

### 4.1 Published Baselines

In this section, we present our reimplementation of two different systems as our published baseline, a graph-based one (following McDonald et. al., 2005) and a transition-based one (following Chen and Manning, 2014). 

#### 4.1.1 Graph-based approach

Following the graph-based approach developed by McDonald et. al.(2005), we factor the score of a dependency tree as the sum of the scores of all edges in the tree. And the score of each edge is the dot product between feature representation of the edge (i.e. [embedding of word1, pos of word1, embedding of word2, pos of word2]) and a weight vector (parameter). We use Margin Infused Relaxed Algorithm(MIRA) algorithm to learn the parameter by narrowing the score of true denpendency tree with current highest score. The algorithm achieved a UAS score of 69.36% on the test set, which has doubled the all-right baseline score.



#### 4.1.2 Transition-based approach

In addtion to the graph-based method, we also reimplemented a transition-based model following the work of the Stanford NLP group (Chen and Manning, 2014), where a parse tree is represented in terms of a series of configurations. A configuration C = (s; b; A) consists of a stack s, a buffer b, and a set of dependency arcs A. For the initial configuration, we have a stack with <Root> and a buffer full of all the words with the arc set A empty. As the parsing proceeds, words are moved from the buffer to the stack, and appropriate actions (LEFT-arc, RIGHT-arc or SHIFT) are taken in order to produce corresponding arcs. For the terminal configuration, we are left with a stack with one token <Root> still, an empty buffer, and a set A of all arcs produced during parsing.
    
The Stanford system is known to be the first algorithm to implement transition-based parsing with neural network. Our reimplementation has a UAS of 77.23% on the test set. In comparison to the original paper, there is still a gap between the performance of our system and theirs, since the latter achieved a UAS of 88.00% on English data. It could be attributable to the fact that our implementation of the activation function (i.e. the Cube function, as described in their paper) has failed to yeild sensible outputs, and consequently we used a more advanced architecture (Bi-LSTM) instead.

Bi-LSTM is the most popular method for sequence to sequence model. In the case of an LSTM, for each element in the sequence, there is a corresponding hidden state， which in principle can contain information from arbitrary points earlier in the sequence. We can use the embedded transition states as inputs and use the output to predict the transitions. 

Another reason for the gap between our model's performance and the original paper is that the dataset that the authors worked on is the English Penn Treebank (over 40,000 sentences), whereas we are using a relatively smaller dataset provided by the Universal Dependency Project (16,622 senteneces in total). These might be potential reasons that the results are not quite comparable.

### 4.2 Extensions

#### 4.2.1 Graph-based model

##### Biaffine transformation

The first and only extension we tried for graph-based model is the use of biaffine transformation, the intuition of which is from Dozat et. al. (2016, 2017). Biaffine transformation is a second order method compared with linear model. The advantage is that it can represent the interactions of two nodes of a edge. Recent paper like c2l2 from Cornell, graph-based model from Stanford achieve great improvements. Also if we combine this method with word/char embedding, this model can kind of automatically give good feature representation, and do not need feature engineering like previous methods. 

The performance acheived by the extened baseline is as follows:

|Model | UAS on test set |
| -------- | --------  |
| Baseline    | 69.36%   |
| Biaffine transformation | 75.11% |
_Table1 - Performance of different versions of graph-based models (with/without extension) on the test set_

#### 4.2.2 Transition-based model

##### Convolutional architecture
The second extension we tried is to use CNN to reduce variance in frequency of input features before passing it to biLSTM layers. This extension is inspired by *Convolutional Neural Networks for Sentence Classification*. In sentence and text classification, CNN is used to extract features from more dimensions and produce a higher order representation before passing it to LSTM layer.

In our method, the features are the top 20 words from stack and buffer in each cofiguration and their corresponding POS tags. We treat each word/POS tag as one feature and represent the features in word embeddings and POS tag embeddings with dimension 100 and 17 respectively. In this way, we will have a fixed embedding matrix for both word features and POS tag features, which is important for CNN. Both word embedding matrix(20\*100) and POS tag matrix(20\*17), will pass 2 convolutional layers with a max pooling layer for each. Then with a linear dimension reduction layer, the output will be maped in size 1\*200 and 1\*17 for word embedding matrix and POS tag matrix. Then these output will be concatenated together and be treated as the input for the LSTM layer. 

However, the results for using cnn is not good. The reson should be that in text classification, we are using cnn since there are similar words and expression in the text if you have the same topic. Thus cnn can capture similar features in each text with higher dimension representation. 

But in the dependency parsing, we are finding the dependency tree, which is like the rule behind the words, and the words themselves are not necessarily similar to each other. So the cnn features extracted from the embeddings will not give effective information. 


##### POS embeddings
The third extension we tried is to use POS embeddings as features. Namely, we trained 20-dimensional POS embeddings based on the sentences from the training set, and used them as part of the input features to our bi-LSTM network. The intuition is to capture potential contextual information encoded in POS tag sequences, instead of representing every tag as an one-hot vector.

The outcome of adding the extension turned out not satisfactory though. With the same number of training epochs (i.e. 10 epochs), it achieved a UAS score of 74.89%. Though the performance surpassed the baseline, it's not better than the original bi-LSTM model with one-hot POS vectors.


##### Lemmatization

We also tried to make use of lemmatization as an extension of the baseline. Specifically, on top of the currently best-performing bi-LSTM model, we used word lemmas instead of word forms provided in the CoNLLU data, and represented the lemmas using word embeddings subsequently as the part of the input to our neural network. We expect that performing lemmatization would lower the proportion of out-of-vocabulary (OOV) words arising from inflection or derivation, and thus generate more effective embeddings. 

The performance of the model with lemmatization added as an extension is 76.92% on the test set.


##### Summary

In comparison, the performance of all the different versions of transition-based model is summrized in the following table:

|Model | UAS on test set |
| -------- | --------  |
| Baseline (Bi-LSTM)   | 77.23%  |
| Bi-LSTM + CNN | 75.11%   |
| Bi-LSTM + POS embedding    | 74.89%   |
| Bi-LSTM + lemmatization | 76.92%  |
_Table2 - Performance of different versions of transition-based models (with different extension) on the test set_

As is seen from the table, though we tried multiple types of extension, the best performance is achieved by the relatively simple version still: Bi-LSTM with 77.23% UAS on test set.

## 5. Conclusions

In this project, we explored the task of dependency parsing on a monolingual corpus, using three different approaches: an intuitive majority baseline, a graph-based model, and a transition-based model. With the initial UAS of 30% acheived by the simple baseline, we are then able to double the baseline performance with the reimplementation of two published models following the graph-based or transition-based paradigm. On the basis of that, we experimented with a set of extensions from the perspective of feature engeneering and architecture manipulation of the network, among which, unfortunately, only a few turned out to work satisfactorily. Eventually, the optimal parser that we have is the Bi-LSTM transition-based model, which ahieved a UAS of 77.23% on the test set.

## Acknowledgements
We would like to show our gratitude to the TA, Stephen Mayhew, for all his supportive feedback during the course of this research.

## References
1. Chen, D., & Manning, C. (2014). A fast and accurate dependency parser using neural networks. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 740-750).
2. Dozat, T., & Manning, C. D. (2016). Deep biaffine attention for neural dependency parsing. arXiv preprint arXiv:1611.01734.
3. Dozat, T., Qi, P., & Manning, C. D. (2017). Stanford's Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task. Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, 20-30.
4. Garcia, M., & Gamallo, P. (2017). A rule-based system for cross-lingual parsing of Romance languages with Universal Dependencies. Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, 274-282.
5. R. McDonald, F. Pereira. (2005).  Non-projective Dependency Parsing using Spanning Tree Algorithms. In Proc. of the Joint Conf. on Human Language Technology and Empirical Methods in Natural Language Processing (HLT/EMNLP).
6. Shi, T., Wu, F. G., Chen, X., & Cheng, Y. (2017). Combining global models for parsing Universal Dependencies. Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, 31-39.
7. Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
