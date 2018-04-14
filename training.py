from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

##Log all errors to the console
tf.logging.set_verbosity(tf.logging.INFO)

#Initialize the reward array
rewards = [0, 0]
num_of_beads = 48
total_wins = [0, 0]

# Game episodes
num_of_episodes = 5000
epilson = 0.1
gamma = 0.99

# ## A four layer NN function to calculate q-values
# ## One input layer, two hidden layers with RELU activations and one output layer
# def q_values(state):
#
#   ##Initializing tf variables for linear model
#   b = tf.Varaiable(tf.zeros((6,)))
#   W = tf.Variable(tf.random_uniform((1, 2), - 1, 1))
#   x = tf.placeholder(tf.int32, (2, 6))
#
#   #Input layer of DNN
#   input_layer = tf.matmul(x, W) + b
#   #Hidden layer #1
#   hidden_layer1 = tf.layers.dense(
#       inputs=input_layer,
#       units=6,
#       activation=tf.nn.relu)
#
#   #Hidden layer #2
#   hidden_layer2 = tf.layers.dense(
#       inputs=hidden_layer1,
#       units=6,
#       activation=tf.nn.relu)
#
#   #Output layer of logits
#   q_values = tf.layers.dense(inputs=hidden_layer2, units=6, activation=None)
#
#   return q_values

def training(playerOneId, playerTwoId):
    ##Initializing tf variables for linear model
    W = tf.Variable(tf.random_uniform((1, 2), - 1, 1))
    x = tf.placeholder(tf.float32, (2, 6))
    global rewards

    # Input layer of DNN
    input_layer = tf.matmul(W, x)

    # Hidden layer #1
    hidden_layer1 = tf.layers.dense(
        inputs=input_layer,
        units=6,
        activation=tf.nn.relu)

    # Output layer of logits
    q_values = tf.layers.dense(inputs=hidden_layer1, units=6, activation=None)

    print(q_values)

    #Define the loss to minimize and pass it to gradient descent optimizer
    next_q_values = tf.placeholder(shape=[6, 1], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_q_values - q_values))


    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    #Initialize the board to start the game
    board = np.full((2, 6), 4, dtype=float)

    for i in range(num_of_episodes):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        while(num_of_beads > 0):
            q_values_playerOne = sess.run(q_values, {x : board})
            playerOne_q_values = q_values_playerOne[0]
            playerOne_action = max_index(playerOne_q_values)
            print(playerOne_action)
            board = emulator(playerOneId, rewards, playerOne_action, board)
            q_values_playerTwo = sess.run(q_values, {x : board})

            playerTwo_q_values = q_values_playerTwo[0]
            playerTwo_action = max_index(playerTwo_q_values)
            print(playerTwo_action)
            board = emulator(playerOneId, rewards, playerTwo_action, board)

            current_loss = tf.reduce_sum(tf.square(playerTwo_q_values - playerOne_q_values))


            #Terminal step rewards is just rewards with no future reward because there is no episode after the terminal
            #episode.
            sess.run(train_step, {loss : current_loss.eval(session=sess)})

            #print step summary
            print("Training step - " + train_step + "\n" + "Loss - " + current_loss + "\n")

        #End of episode

        #Increment the win board
        if(rewards[0] < rewards[1]):
            total_wins[playerTwoId] += 1
        else:
            total_wins[playerOneId] += 1

        #re-initialize the oware board
        board = np.full((2, 6), 4, dtype=int)

        #re-initialize the reward array
        rewards = [0, 0]

def printResult(playerOneId, playerTwoId):
    print("The agents played " + num_of_episodes + " Agent-1 " + "won " + total_wins[playerOneId] + "Agent-2 won " + total_wins[playerTwoId])

def max_index(logits):
    max_index = 0
    i = 0
    for x in np.nditer(logits):
        if(logits[i + 1] > logits[i]):
            max_index = i + 1

        if(i == 4):
            break
        i = i + 1

    return max_index;

#Emulating playing the game of oware by taking playerId, reward array, action and current board configuration
def emulator(player, rewards, action, currState):
    print(currState.shape)
    beads = currState[player][action]
    currState[player][action] = 0
    global num_of_beads

    i = player;
    j = action;



    while (beads > 0):
        ##iterate over the board in anti-clockwise direction

        if (i == 0):
            if (j == 5):
                i = 1
            else:
                j = j + 1

        else:
            if (j == 0):
                i = 0
            else:
                j = j - 1;

        currState[i][j] = currState[i][j] + 1
        beads = beads - 1

        if (beads == 0):
            if (currState[i][j] == 2):
                currState[i][j] = 0
                rewards[player] += 2
                num_of_beads -= 2
                break

            elif (currState[i][j] == 4):
                currState[i][j] = 0
                rewards[player] += 4
                num_of_beads -= 4
                break

            elif (currState[i][j] == 1):
                break

            else:
                beads = currState[i][j]
                currState[i][j] = 0
    return currState



def main(unused_argv):
    training(0, 1)
    printResult(0, 1)

if __name__ == "__main__":
  tf.app.run()
