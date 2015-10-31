print(__doc__)

from sklearn.svm import LinearSVC, SVC
import BatchReader

reader = BatchReader.inputs()
array = reader.getNPArray(5990)

X_digits = array[0]
y_digits = array[1]

n_samples = len(X_digits)

X_train = X_digits[:.9*n_samples]
y_train = y_digits[:.9*n_samples]
X_test = X_digits[.9*n_samples:]
y_test = y_digits[.9*n_samples:]

ovasvc = LinearSVC(tol=1e-3, multi_class='crammer_singer')

print('Linear SVC (OVA) score: %f' % ovasvc.fit(X_train, y_train).score(X_test,y_test))