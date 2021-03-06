import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.
        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        weight = self.get_weights()
        return (nn.DotProduct(x, weight))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        num = nn.as_scalar(self.run(x))

        if num >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        mistakes = True
        #run until no mistakes
        while mistakes:
            mistakes = False
            #iterate data set
            for x, y in dataset.iterate_once(1):
                #perdiction is not scalay for y
                if self.get_prediction(x) != nn.as_scalar(y):
                    #update teh weights
                    self.get_weights().update(x, nn.as_scalar(y))
                    mistakes = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #initialize parameters
        self.w0 = nn.Parameter(1, 250)
        self.w1 = nn.Parameter(250, 1)
        self.b0 = nn.Parameter(1, 250)
        self.b1 = nn.Parameter(1, 1)

        self.bath_size = 1
        self.learningRate = -0.015
        self.threshold = 0.02

    def run(self, x):
        """g
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        l0in = nn.Linear(x, self.w0)
        l0mid = nn.AddBias(l0in, self.b0)
        l0out = nn.ReLU(l0mid)
        l1in = nn.Linear(l0out, self.w1)
        l1mid = nn.AddBias(l1in, self.b1)
        return (l1mid)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return (nn.SquareLoss(self.run(x), y))

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        train = True
        #train model
        while train:
            #get the loss
            lost = self.get_loss(nn.Constant(dataset.x),
                                 nn.Constant(dataset.y))
            #if we break threashold then stop
            if nn.as_scalar(lost) <= self.threshold:
                break
            
            #calc gradients
            gw0, gb0, gw1, gb1 = nn.gradients(
                lost, [self.w0, self.b0, self.w1, self.b1])

            #update the parameters
            self.w0.update(gw0, self.learningRate)
            self.w1.update(gw1, self.learningRate)
            self.b0.update(gb0, self.learningRate)
            self.b1.update(gb1, self.learningRate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.
    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.
    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.W2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.W3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)
        self.W4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.
        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        second = nn.ReLU(nn.AddBias(nn.Linear(first, self.W2), self.b2))
        third = nn.ReLU(nn.AddBias(nn.Linear(second, self.W3), self.b3))
        fourth = nn.AddBias(nn.Linear(third, self.W4), self.b4)

        return (fourth)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return (nn.SoftmaxLoss(self.run(x), y))

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.97:
            for x, y in dataset.iterate_once(1000):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                
                for i in range(8):
                    self.params[i].update(gradient[i], -0.99)
