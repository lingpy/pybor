"""
Bag of sound approaches for borrowing detection.
"""

# Import Python standard libraries
from collections import defaultdict

# Import 3rd-party libraries
from sklearn import svm
import attr


@attr.s
class BagOfSounds:
    """
    Use an SVM classifier to try to find borrowed words.

    Notes
    -----
    This is following the basic structure of SVMs as implemented in
    scikitlearn: https://scikit-learn.org/stable/modules/svm.html
    """

    data = attr.ib()
    kernel = attr.ib(default="linear")

    def __attrs_post_init__(self):

        self.sounds = defaultdict(int)
        # iterate over words and make a first list of all sounds
        for _, word, status in self.data:
            # assign word to the bag of sounds
            for sound in word:
                self.sounds[sound] += 1
        self.features = sorted(self.sounds, key=lambda x: x, reverse=True)
        # second run, convert to the format needed by the SVM code
        # we use variables X and y from the SVM tutorial
        self.X, self.y = [], []
        for _, word, status in self.data:
            features = []
            for sound in self.features:
                if sound in word:
                    features += [1]
                else:
                    features += [0]
            self.X += [features]
            self.y += [status]

        # self.X = np.array(self.X)
        self.clf = svm.SVC(kernel=self.kernel)
        self.clf.fit(self.X, self.y)

    def predict(self, word):
        """
        Predict if a word is borrowed or not.
        """
        # convert to features
        features = []
        for sound in self.features:
            if sound in word:
                features += [1]
            else:
                features += [0]

        return self.clf.predict([features])

    def predict_data(self, data):
        """
        Predict for a range of words.
        """
        out = []
        for idx, word in data:
            out += [[idx, word, self.predict(word)]]

        return out
