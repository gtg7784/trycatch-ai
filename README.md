# trycatch-ai
2019 emotion summer vacation project - trycatch ai (reinforcement learning, Deep Q Networks)

# result
[![result.mov]](https://github.com/gtg7784/trycatch-ai/blob/master/result/result.mov)

# 코드 설명

### train.py

```
# -*- coding: utf-8 -*-
```
utf-8 형식으로 인코딩합니다.

```
import tensorflow as tf
import numpy as np
import random
import math
import os
```
텐서플로우 및 학습에 필요한 라이브러리를 불러옵니다.

```
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
```
학습에 필요한 설정값을 선언합니다.

epsilon-Greedy 기법에 사용할 초기 입실론값 - epsilon  
최소 입실론값 - epsilonMinimumValue  
가능한 행동의 개수 (좌, 우, 가만히 있기) - num_actions  
학습에 사용할 반복 횟수 - num_epochs  
은닉층의 노드 개수 - hidden_size  
리플리에 메모리의 최대 크기 - maxMemory  
배치 사이즈 - batch_size  
게임의 크기 - gridSize  
Discount Factor - discount  
Learning Rate - learning_rate  

```
def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;
```
s(start)와 e(end) 사이의 랜덤한 값을 리턴하는 randf 함수를 정의합니다.

```
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
```
Deep Q Network를 만드는 build_DQN 함수를 정의합니다.  
DQN은 입력층(input layer)으로 게임의 현재 상태 s(10 * 10 Grid = 100)를 입력받아서, 128개의 노드를 가지고 있는 은닉층(hidden layer)을 2개 거쳐서 현재 상태에서 취할수 있는 각각의 행동에 대한 Q(s, a)(3개) 예측값을 출력층(output layer)에서 출력합니다.

```
x = tf.placeholder(tf.float32, shape=[None, state_size])
y = tf.placeholder(tf.float32, shape=[None, num_actions])

y_pred = build_DQN(x)
```
인풋인 현재 상태(10 * 10 Grid = 100)와 타겟 Q*(s, a)값을 입력 받을 플레이스홀더 x, y를 선언하고, build_DQN 함수를 사용하여 DQN 모델을 만듭니다.

```
loss = tf.reduce_sum(tf.square(y-y_pred)) / (2*batch_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```
MSE 손실 함수와 옵티마이저를 정의합니다.

```
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
```
게임의 플레이 환경을 만드는 CatchEnvironment 클래스를 정의합니다. 각각의 함수별로 수행하는 기능은 다음과 같습니다.
1. ` __init__` : 클래스 생성시 호출되는 생성자로써 게임의 상태값들을 
2. `observe` : drawState를 호출해서 생성된 관찰 결과를 리턴합니다.
3. `drawState` : 상태값에 따라 캔버스에 과일과 바구니를 그립니다.
4. `reset` : 게임을 초기상태로 리셋합니다. initialFruitColumn, initialBucketPosition을 랜덤한 값으로 초기화해서 과일을 캔버스 가로축 최상단의 랜덤한 위치, 바구니를 캔버스 가로축 최하단의 랜덤한 위치로 할당합니다.
5. `getState` : 게임의 현재 상태를 불러옵니다. 과일은 몇 번쨰 세로축에 있고, 얼만큼 떨어져서 몇번째 가로축에 있는지, 바구니는 몇번째 세로축에 있는지를 리턴합니다.
6. `getReward` : 에이전트가 취한 행동에 대한 보상을 얻습니다. 만약 과일이 바닥에 닿았을 때, 바구니가 과일을 받아내면 1의 reward를 주고, 받아내지 못하면 -1의 reward를 줍니다. 그리고 과일이 바닥에 닿지 않을 때에는 0을 줍니다.
7. `isGameOver` : 게임이 끝났는지를 체크합니다. 과일이 바닥에 닿으면 게임을 종료합니다.
8. `updateState` : 에이전트의 action에 따라 바구니의 위치를 업데이트 하고, 한 스텝 시간이 흘러서 과일이 한 칸씩 떨어지는 상태의 업데이트를 진행합니다.
9. `act` : 에이전트가 행동을 취해서 상태를 업데이트하고, 해당 행동에 대한 reward와 게임 종료 유무를 체크해서 리턴합니다.

```
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
```
리플레이 메모리를 구현한 ReplayMemory 클래스를 정의합니다. 각각의 함수별로 수행하는 기능을 정리하면 다음과 같습니다.

1. `__init__` : 클래스 생성시 호출되는 생성자로써 리플레이 메모리의 상태값을 초기화합니다.
2. `remember` : 현재 경험(S, A, R(s, s'), s')을 리플레이 메모리에 저장합니다.
3. `getBatch` : 리플레이 메모리에서 랜덤 샘플링을 통해 임의의 기억을 가져오고, 해당 기억의 현재 상태값과 다음 상태값을 DQN에 넣고 Q* = reward + discount(gamma) * max_a' Q(s',a') 식을 계산해서 얻은 타겟 Q값을 배치크기만큼 묶어서 리턴합니다.

```
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
```
main 함수입니다. CatchEnvironment와 ReplayMemory 클래스를 선언하고, 학습된 파라미터를 저장할 saver를 선언합니다. 세션을 열어서 변수에 초기값들을 선언하고,초기 action의 Q 값을 -9999로 초기화합니다. epsilon 확률로 랜덤한 행동을 하고, (1 - epsilon) 확률만큼 DQN을 이용해서 각각의 행동에 대한 Q값을 예측하고 가장 Q값이 큰 최적의 행동을 합니다.  
epsilon-Greedy 기법을 이용해서 epsilon 값을 점차 갑소시켜, 에이전트가 학습 초반에는 랜덤한 행동을 많이 하고, 학습이 진행될수록 최적의 행동을 하도록 유도합니다. 또한 현재 경험(S, A, R(s, s'), s')을 리플레이 메모리에 저장하고, 리플레이 메모리에서 랜덤 샘플링을 통해서 임의의 배치를 불러와서 최적화를 진행합니다. 지정된 횟수만큼 반복이 끝나면 학습된 파라미터를 model.ckpt 체크포인트 파일로 저장합니다.

```
if __name__ == '__main__':
  tf.app.run()
```
tf.app.run() 함수를 실행합니다.

-----

### visualize.ipynb

```
# -*- coding: utf-8 -*-
```
utf-8 형식으로 인코딩합니다.

```
%matplotlib
%matplotlib inline

from train import *
from IPython import display
import matplotlib.patches as patches
import pylab as pl
import time
import tensorflow as tf
import os
```
필요한 라이브러리를 임포트합니다.

```
gridSize = 10
maxGames = 100
env = CatchEnvironment(gridSize)
winCount = 0
loseCount = 0
numberOfGames = 0
```
필요한 설정값들을 정의합니다. maxGames 횟수만큼 DQN 에이전트가 게임을 플레이합니다.

```
ground = 1
plot = pl.figure(figsize=(12,12))
axis = plot.add_subplot(111, aspect='equal')
axis.set_xlim([-1, 12])
axis.set_ylim([0, 12])
```
화면을 그리기 위한 설정값들을 정의합니다.

```
saver = tf.train.Saver()
```
model.ckpt 파일로 저장한 학습된 파라미터를 읽어오기 위해서 tf.train.Saver() API를 선언합니다.

```
def drawState(fruitRow, fruitColumn, basket, gridSize):
  fruitX = fruitColumn 
  fruitY = (gridSize - fruitRow + 1)
  statusTitle = "Wins: " + str(winCount) + "  Losses: " + str(loseCount) + "  TotalGame: " + str(numberOfGames)
  axis.set_title(statusTitle, fontsize=30)
  for p in [
    patches.Rectangle(
        ((ground - 1), (ground)), 11, 10,
        facecolor="#000000"
    ),
    patches.Rectangle(
        (basket - 1, ground), 2, 0.5,
        facecolor="#FF0000"
    ),
    patches.Rectangle(
        (fruitX - 0.5, fruitY - 0.5), 1, 1,
        facecolor="#0000FF"       # Blue
    ),   
    ]:
      axis.add_patch(p)
  display.clear_output(wait=True)
  display.display(pl.gcf())
```
현재 상태를 화면에 그리기 위한 drawState 함수를 정의합니다. 과일은 파란색, 바구니는 빨간색, 배경은 검은색 네모로 표현합니다. 승리 횟수, 패배 횟수, 전체 게임 횟수를 화면 상단에 출력합니다.

```
with tf.Session() as sess:    
  saver.restore(sess, os.getcwd()+"/model.ckpt")
  print('파라미터를 불러왔습니다!')

  while (numberOfGames < maxGames):
    numberOfGames = numberOfGames + 1
     
    isGameOver = False
    fruitRow, fruitColumn, basket = env.reset()
    currentState = env.observe()
    drawState(fruitRow, fruitColumn, basket, gridSize)

    while (isGameOver != True):
      q = sess.run(y_pred, feed_dict={x: currentState})
      action = q.argmax()

      nextState, reward, gameOver, stateInfo = env.act(action)    
      fruitRow = stateInfo[0]
      fruitColumn = stateInfo[1]
      basket = stateInfo[2]
     
      if (reward == 1):
        winCount = winCount + 1
      elif (reward == -1):
        loseCount = loseCount + 1

      currentState = nextState
      isGameOver = gameOver
      drawState(fruitRow, fruitColumn, basket, gridSize)
    
      time.sleep(0.003)

display.clear_output(wait=True)
```
세선을 열어서 그래프를 실행합니다. model.ckpt 파일로부터 저장된 DQN 파라미터를 읽어오고, 현재 상태를 DQN의 입력값으로 넣고 구한 Q값 중 가장 큰 Q값을 가지는 행동을 취합니다.
