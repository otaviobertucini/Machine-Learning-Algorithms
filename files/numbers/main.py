from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
mnist = datasets.load_digits()

X, y = mnist['data'], mnist['target']

y_test = (y == 0)
print(y_test)

# some_digit = X[0]
# some_digit_image = some_digit.reshape(8, 8)
#
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
#
# plt.savefig('number.png')