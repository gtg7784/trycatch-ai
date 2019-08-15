# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import math
import os

epsilon = 1
epsilonMinimumValue = .001
num_actions = 3
num_epochs = 2000
hidden_size = 128
maxMemory = 500
batch_size = 50
gridSize = 10
state_size = gridSize * gridSize 
discount = 0.9
learning_rate = 0.2	

def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

def build_DQN(x):
  W1 = tf.Variable(tf.truncated_normal(shape=[state_size, hidden_size], stddev=1.0 / math.sqrt(float(state_size))))
  b1 = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.01))  
  H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
  W2 = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size],stddev=1.0 / math.sqrt(float(hidden_size))))
  b2 = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.01))
  H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)
  W3 = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_actions],stddev=1.0 / math.sqrt(float(hidden_size))))
  b3 = tf.Variable(tf.truncated_normal(shape=[num_actions], stddev=0.01))
  output_layer = tf.matmul(H2_output, W3) + b3

  return tf.squeeze(output_layer)

x = tf.placeholder(tf.float32, shape=[None, state_size])
y = tf.placeholder(tf.float32, shape=[None, num_actions])

y_pred = build_DQN(x)

loss = tf.reduce_sum(tf.square(y-y_pred)) / (2*batch_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

class CatchEnvironment():
  def __init__(self, gridSize):
    self.gridSize = gridSize
    self.state_size = self.gridSize * self.gridSize
    self.state = np.empty(3, dtype = np.uint8) 

  def observe(self):
    canvas = self.drawState()
    canvas = np.reshape(canvas, (-1,self.state_size))
    return canvas

  def drawState(self):
    canvas = np.zeros((self.gridSize, self.gridSize))
    canvas[self.state[0]-1, self.state[1]-1] = 1  
    canvas[self.gridSize-1, self.state[2] -1 - 1] = 1
    canvas[self.gridSize-1, self.state[2] -1] = 1
    canvas[self.gridSize-1, self.state[2] -1 + 1] = 1    
    return canvas        

  def reset(self): 
    initialFruitColumn = random.randrange(1, self.gridSize + 1)
    initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
    self.state = np.array([1, initialFruitColumn, initialBucketPosition]) 
    return self.getState()

  def getState(self):
    stateInfo = self.state
    fruit_row = stateInfo[0]
    fruit_col = stateInfo[1]
    basket = stateInfo[2]
    return fruit_row, fruit_col, basket

  def getReward(self):
    fruitRow, fruitColumn, basket = self.getState()
    if (fruitRow == self.gridSize - 1):  
      if (abs(fruitColumn - basket) <= 1): 
        return 1
      else:
        return -1
    else:
      return 0

  def isGameOver(self):
    if (self.state[0] == self.gridSize - 1): 
      return True 
    else: 
      return False 

  def updateState(self, action):
    move = 0
    if (action == 0):
      move = -1
    elif (action == 1):
      move = 0
    elif (action == 2):
      move = 1
    fruitRow, fruitColumn, basket = self.getState()
    newBasket = min(max(2, basket + move), self.gridSize - 1)
    fruitRow = fruitRow + 1 
    self.state = np.array([fruitRow, fruitColumn, newBasket])

  def act(self, action):
    self.updateState(action)
    reward = self.getReward()
    gameOver = self.isGameOver()
    return self.observe(), reward, gameOver, self.getState()


class ReplayMemory:
  def __init__(self, gridSize, maxMemory, discount):
    self.maxMemory = maxMemory
    self.gridSize = gridSize
    self.state_size = self.gridSize * self.gridSize
    self.discount = discount
    canvas = np.zeros((self.gridSize, self.gridSize))
    canvas = np.reshape(canvas, (-1,self.state_size))
    self.inputState = np.empty((self.maxMemory, 100), dtype = np.float32)
    self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
    self.nextState = np.empty((self.maxMemory, 100), dtype = np.float32)
    self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
    self.rewards = np.empty(self.maxMemory, dtype = np.int8) 
    self.count = 0
    self.current = 0

  def remember(self, currentState, action, reward, nextState, gameOver):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.inputState[self.current, ...] = currentState
    self.nextState[self.current, ...] = nextState
    self.gameOver[self.current] = gameOver
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.maxMemory

  def getBatch(self, y_pred, batch_size, num_actions, state_size, sess, X):
    memoryLength = self.count
    chosenBatchSize = min(batch_size, memoryLength)

    inputs = np.zeros((chosenBatchSize, state_size))
    targets = np.zeros((chosenBatchSize, num_actions))

    for i in range(chosenBatchSize):
      randomIndex = random.randrange(0, memoryLength)
      current_inputState = np.reshape(self.inputState[randomIndex], (1, 100))
      target = sess.run(y_pred, feed_dict={X: current_inputState})
      current_nextState = np.reshape(self.nextState[randomIndex], (1, 100))
      nextStateQ = sess.run(y_pred, feed_dict={X: current_nextState})      
      nextStateMaxQ = np.amax(nextStateQ)

      if (self.gameOver[randomIndex] == True):
        target[self.actions[randomIndex]] = self.rewards[randomIndex]
      else:
        target[self.actions[randomIndex]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

      inputs[i] = current_inputState
      targets[i] = target

    return inputs, targets

def main(_):
  print("트레이닝을 시작합니다.")

  env = CatchEnvironment(gridSize)

  memory = ReplayMemory(gridSize, maxMemory, discount)

  saver = tf.train.Saver()
  
  winCount = 0
  with tf.Session() as sess:   
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs+1):
      err = 0
      env.reset()

      isGameOver = False

      currentState = env.observe()
            
      while (isGameOver != True):
        action = -9999
        global epsilon
        if (randf(0, 1) <= epsilon):
          action = random.randrange(0, num_actions)
        else: 
          q = sess.run(y_pred, feed_dict={x: currentState})   
          action = q.argmax()

        if (epsilon > epsilonMinimumValue):
          epsilon = epsilon * 0.999
        
        nextState, reward, gameOver, stateInfo = env.act(action)
            
        if (reward == 1):
          winCount = winCount + 1

        memory.remember(currentState, action, reward, nextState, gameOver)
        
        currentState = nextState
        isGameOver = gameOver
                
        inputs, targets = memory.getBatch(y_pred, batch_size, num_actions, state_size, sess, x)
        
        _, loss_print = sess.run([optimizer, loss], feed_dict={x: inputs, y: targets})  
        err = err + loss_print

      print("반복(Epoch): %d, 에러(err): %.4f, 승리횟수(Win count): %d, 승리비율(Win ratio): %.4f" % (i, err, winCount, float(winCount)/float(i+1)*100))
    print("트레이닝 완료")
    save_path = saver.save(sess, os.getcwd()+"/model.ckpt")
    print("%s 경로에 파라미터가 저장되었습니다" % save_path)

if __name__ == '__main__':
  tf.app.run()