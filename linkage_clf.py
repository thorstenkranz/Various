import copy

import mvpa
from mvpa.suite import CrossValidatedTransferError, TransferError, ConfusionMatrix, TreeClassifier, Classifier, NFoldSplitter
from mvpa.clfs.libsvmc import SVM
from mvpa.misc.state import ClassWithCollections
import numpy as np
import hcluster

class LinkageClassifier(Classifier):
    """Linkage classifier using hierarchical clustering.
    For training-dataset it does a cross-validation. From the resulting confusion matrix, 
    the simililaity between labels is evaluated. From this, a TreeClassifier is created and trained."""
    def __init__(self,clf,splitter=None, *args,**kwargs):
        Classifier.__init__(self,*args,**kwargs)
        self._clf = clf
        self._confusion = None
        self._dataset = None
        self.ulabels = None
        if splitter == None:
            self._splitter = NFoldSplitter()
        else:
            self._splitter = splitter
        self.linkage = None
        self._tree = None
        self._tree_clf = None

    def _train(self, dataset):
        self._dataset = dataset
        self.ulabels=self._dataset.uniquelabels
        # Do cross-validation for normal classifier
        self.cvterr = CrossValidatedTransferError(TransferError(self._clf),self._splitter,enable_states=["confusion"])
        self.cvterr(self._dataset)
        # From the confusion matrix, calculate linkage and tree-structure
        # First prepare distance matrix from confusion matrix
        dist = self.cvterr.confusion.matrix
        dist = (dist+dist.T)/2 # Distance must be symmetric (property of a norm)
        dist = dist.max()-dist # Kind of inversion. High values in confusion -> similar -> small distance
        dist -= np.diag(np.diag(dist)) # Distance to self must be zero -> make diagonal elements zero
        # Calculate linkage matrix
        self.linkage = hcluster.linkage(hcluster.squareform(dist))
        # Build tree and according TreeClassifier
        self.tree = hcluster.to_tree(self.linkage)
        self._tree_clf = self.build_tree_classifier_from_linkage_tree(self.tree)[0]
        self._tree_clf.train(self._dataset)
        #print "Trained on", self.ulabels

    def build_tree_classifier_from_linkage_tree(self, tree):
        # Here SVM is only used as it can do single-class classification (pseudo)
        if tree.left.is_leaf() and tree.right.is_leaf():
            return (self._clf.clone(),[tree.left.id, tree.right.id])
        elif tree.left.is_leaf():
            clf1, ids1 = (SVM(), [tree.left.id])
            clf2, ids2 = self.build_tree_classifier_from_linkage_tree(tree.right)
            return (TreeClassifier(self._clf.clone(),
                                    {"c%02i"%tree.left.id:([self.ulabels[i] for i in ids1],clf1), "c%02i"%tree.right.id:([self.ulabels[i] for i in ids2],clf2)}), 
                    ids1+ids2)
        elif tree.right.is_leaf():
            clf1, ids1 = self.build_tree_classifier_from_linkage_tree(tree.left)
            clf2, ids2 = (SVM(), [tree.right.id])
            return (TreeClassifier(self._clf.clone(),
                                    {"c%02i"%tree.left.id:([self.ulabels[i] for i in ids1],clf1), "c%02i"%tree.right.id:([self.ulabels[i] for i in ids2],clf2)}), 
                    ids1+ids2)
        else:
            clf1, ids1 = self.build_tree_classifier_from_linkage_tree(tree.left)
            clf2, ids2 =  self.build_tree_classifier_from_linkage_tree(tree.right)
            return (TreeClassifier(self._clf.clone(),
                                    {"c%02i"%tree.left.id:([self.ulabels[i] for i in ids1],clf1), "c%02i"%tree.right.id:([self.ulabels[i] for i in ids2],clf2)}), 
                    ids1+ids2)
    
    def untrain(self):
        self._tree_clf = None
        super(LinkageClassifier,self).untrain()

    def _predict(self,data):
        if self._tree_clf != None:
            return self._tree_clf.predict(data)
        else:
            raise ValueError("Classifier wasn't yet trained, so cannot predict.")

    def dendrogram(self):
        #import pylab as p
        if not self.linkage == None:
            hcluster.dendrogram(self.linkage,labels=np.unique(self._dataset.labels))


if __name__=="__main__":
    from sys import stdout
    from mvpa.suite import Dataset, FeatureSelectionClassifier, OneWayAnova, FixedNElementTailSelector, SensitivityBasedFeatureSelection, NGroupSplitter, ConfusionMatrix, GNB
    # features of sample data
    print "Generating samples..."
    nfeat = 1000
    nsamp = 200
    ntrain = 90
    goodfeat = 10
    offset = 0.9

    # create the sample datasets
    samp = np.random.randn(nsamp,nfeat)
    for i in range(10):
        samp[i::10,i*goodfeat:(i+1)*goodfeat] += offset

    #assert 1==0

    # create the pymvpa training dataset from the labeled features
    dataset = Dataset(samples=samp[:,:], labels=range(10)*(nsamp/10))

    #clfr1 = FeatureSelectionClassifier(
    #                SVM(),
    #                SensitivityBasedFeatureSelection(
    #                    OneWayAnova(),
    #                    FixedNElementTailSelector(50,mode="select", tail = "upper")
    #                )
    #        )
    clfr1 = GNB()
    clfr2 = LinkageClassifier(clfr1)
    # create patters for the testing dataset
    #patternsPos = Dataset(samples=samp1[ntrain:,:], labels=1)
    #patternsNeg = Dataset(samples=samp2[ntrain:,:], labels=0)
    #testpat = patternsPos + patternsNeg
    #cvterr = CrossValidatedTransferError(TransferError(clfr2),NGroupSplitter(10),enable_states=["confusion"])
    cm1 = ConfusionMatrix()
    cm2 = ConfusionMatrix()
    for i, (d1,d2) in enumerate(NGroupSplitter(10)(dataset)):
        print "Split No.", i, d1,d2
        stdout.flush()
        clfr1.train(d1)
        cm1.add(d2.labels,clfr1.predict(d2.samples))
        clfr2.train(d1)
        cm2.add(d2.labels,clfr2.predict(d2.samples))
        print cm1.stats["ACC"], cm2.stats["ACC"]
    #print cvterr(dataset)
    #print cvterr
