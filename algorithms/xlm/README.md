# extreme-learning-machine
An extreme learning machine (XLM) implemented in Python,
deployed using C++

Extreme learning machines project the input to a random feature space.
(The random projection helps exploit hidden information among the data.)
After the random projection a linear classifier is used.
(Least mean squares optimization using the Moore Penrose psuedo inverse of the matrix.)

Easier to linearly seperate the data after the random projection, amazingly!

But they require a large number of hidden units on large datasets.

Also, this model can't be incrementally trained.

But the idea is really cool and they train super quick.