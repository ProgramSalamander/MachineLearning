from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (list(self.weights), self.bias)

    # 预测输出
    # y = x1*w1 + x2*w2 + ... + xi*wi + b
    def predict(self, input_vec):
        def add(a, b):
            return a + b

        def mul(a):
            return a[0]*a[1]

        return self.activator(
            reduce(add, map(mul, zip(input_vec, self.weights)), 0.0) + self.bias)

    # 多次迭代调整权重,优化感知器的拟合效果
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self.__one_iteration(input_vecs, labels, rate)

    # 一次训练迭代,根据输出更新权重
    def __one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self.__update_weights(input_vec, output, label, rate)

    # 按照感知器规则更新权重
    # w <- w + rate * (label - output) * x
    # b <- b + rate * (label - output)
    def __update_weights(self, input_vec, output, label, rate):
        delta = label - output

        def update(x_w):
            return x_w[1] + delta * rate * x_w[0]

        self.weights = list(map(update, zip(input_vec, self.weights)))
        self.bias += rate * delta


# 激活函数
def f(x):
    return 1 if x > 0 else 0


def train_and_perceptron():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    p = Perceptron(2, f)
    p.train(input_vecs, labels, 100, 0.1)
    return p


and_perceptron = train_and_perceptron()
# 打印训练获得的权重
print(and_perceptron)
# 测试
print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
print('0 and 1 = %d' % and_perceptron.predict([0, 1]))
print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
