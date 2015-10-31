print(__doc__)

from sklearn import neighbors, linear_model
import BatchReader

reader = BatchReader.inputs()
array = reader.getNPArray(7809)

X_digits = array[0]
y_digits = array[1]

n_samples = len(X_digits)

X_train = X_digits[:.9*n_samples]
y_train = y_digits[:.9*n_samples]
X_test = X_digits[.9*n_samples:]
y_test = y_digits[.9*n_samples:]

logistic = linear_model.LogisticRegression()

print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test,y_test))
