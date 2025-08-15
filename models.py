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
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0: 
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        batch_size = 1

        while not converged:
            converged = True 
            for x, y in dataset.iterate_once(batch_size):
                feature = x 
                label = y
                label_guess = self.get_prediction(feature)

                if label_guess > nn.as_scalar(label): 
                    converged = False
                    self.w.update(feature, -1)
                if label_guess < nn.as_scalar(label): 
                    converged = False
                    self.w.update(feature, 1)


                #print(f"label: {nn.as_scalar(label)} guess: {label_guess}")
                
            
#python3 autograder.py -q q1

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_size = 10
        num_layers = 3
        size_down = 0
        final_height = 1 

        self.W = [(nn.Parameter(1, hidden_size), nn.Parameter(1, hidden_size))]

        new_size = hidden_size
        for i in range(1, num_layers-1): 
            last_size = new_size
            new_size -= size_down
            self.W.append((nn.Parameter(last_size, new_size), nn.Parameter(1, new_size)))
    
        self.W.append((nn.Parameter(new_size, final_height), nn.Parameter(1, final_height)))


        print(f"Initilization: {self.W}")
        #print(f"test: {self.W[0]._forward()}")

#python3 autograder.py -q q2

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        final_prediction = x

        for i in range(len(self.W)): 
            w, b = self.W[i][0], self.W[i][1]
            #print(f"input: {final_prediction} weight: {w}")

            lin = nn.Linear(final_prediction, w)
            predicted = nn.AddBias(lin, b)

            if not i == len(self.W)-1: 
                final_prediction = nn.ReLU(predicted)
            else:
                final_prediction = predicted


        return final_prediction

            
#python3 autograder.py -q q2
        

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
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.01

        total_loss = 100 
        while total_loss > 0.01: 
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            params = [param for sublist in self.W for param in sublist]

            gradients = [gradient for gradient in nn.gradients(loss, params)]

            for i in range(len(params)):
                params[i].update(gradients[i], learning_rate)
            
            total_loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))  
            
#python3 autograder.py -q q2


        #print(f"test: {gradients}")



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
        input_size = 784
        hidden_size = 300
        num_layers = 4
        size_down = 100
        final_height = 10

        self.W = [(nn.Parameter(input_size, hidden_size), nn.Parameter(1, hidden_size))]

        new_size = hidden_size
        for i in range(1, num_layers-1): 
            last_size = new_size
            new_size -= size_down
            self.W.append((nn.Parameter(last_size, new_size), nn.Parameter(1, new_size)))
    
        self.W.append((nn.Parameter(new_size, final_height), nn.Parameter(1, final_height)))


        print(f"Initilization: {self.W}")

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

        final_prediction = x

        for i in range(len(self.W)): 
            w, b = self.W[i][0], self.W[i][1]
            #print(f"input: {final_prediction} weight: {w}")

            lin = nn.Linear(final_prediction, w)
            predicted = nn.AddBias(lin, b)

            if not i == len(self.W)-1: 
                final_prediction = nn.ReLU(predicted)
            else:
                final_prediction = predicted


        #print(f"input: {final_prediction}")
                
        return final_prediction

#python3 autograder.py -q q3

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
        prediction = self.run(x)
        return nn.SoftmaxLoss(prediction, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.30
        accuracy = -10000
        last_accuracy = accuracy

        print(f"test: {nn.Constant(dataset.x)}")

        
        while accuracy < 0.98: 
            for x,y in dataset.iterate_once(600): 
                loss = self.get_loss(x, y)
                params = [param for sublist in self.W for param in sublist]
                gradients = [gradient for gradient in nn.gradients(loss, params)]

                for i in range(len(params)):
                    params[i].update(gradients[i], learning_rate)

                accuracy = dataset.get_validation_accuracy()

                if accuracy <= last_accuracy: 
                    learning_rate = min(learning_rate+0.00005, -0.00005)

                last_accuracy = accuracy
                print(f"Current Accuracy: {accuracy} Learning Rate: {learning_rate}")
        

#python3 autograder.py -q q3
#60000

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        input_size = self.num_chars
        hidden_size = 100
        num_layers = 3
        size_down = 0
        final_height = len(self.languages)

        self.W = [(nn.Parameter(input_size, hidden_size), nn.Parameter(1, hidden_size))]

        new_size = hidden_size
        for i in range(1, num_layers-1): 
            last_size = new_size
            new_size -= size_down
            self.W.append((nn.Parameter(last_size, new_size), nn.Parameter(1, new_size)))
    
        self.W.append((nn.Parameter(new_size, final_height), nn.Parameter(1, final_height)))

        print(f"Initilization: {self.W}")
        
#python3 autograder.py -q q4

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        

        w_initial, b_initial = self.W[0][0], self.W[0][1]
        predicted_initial = nn.AddBias(nn.Linear(xs[0], w_initial), b_initial) 
        activation_initial = nn.ReLU(predicted_initial)

        final_prediction = activation_initial

        for letter in range(1, len(xs)): 
            for i in range(1, len(self.W)-1): 
                w, b = self.W[i][0], self.W[i][1] 

                lin = nn.Add(nn.Linear(xs[letter], w_initial), nn.Linear(final_prediction, w))
                predicted = nn.AddBias(lin, b)
                final_prediction = nn.ReLU(predicted)

        w_final, b_final = self.W[-1][0], self.W[-1][1]

        return nn.AddBias(nn.Linear(final_prediction, w_final), b_final) 

#python3 autograder.py -q q4      

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        prediction = self.run(xs)
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        learning_rate = -0.30
        accuracy = -10000
        last_accuracy = accuracy

        
        while accuracy < 0.85: 
            for x,y in dataset.iterate_once(600): 
                loss = self.get_loss(x, y)
                params = [param for sublist in self.W for param in sublist]
                gradients = [gradient for gradient in nn.gradients(loss, params)]

                for i in range(len(params)):
                    params[i].update(gradients[i], learning_rate)

                accuracy = dataset.get_validation_accuracy()

                if accuracy <= last_accuracy: 
                    learning_rate = min(learning_rate+0.00005, -0.00005)

                last_accuracy = accuracy
                #print(f"Current Accuracy: {accuracy} Learning Rate: {learning_rate}")
        




#python3 autograder.py -q q4