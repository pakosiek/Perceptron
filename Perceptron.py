import numpy as np
import matplotlib.pyplot as plt

def losuj_wagi(seed=None): #function to initialize random weights
  if seed:
    np.random.seed(seed)
  return np.random.random(3)*2-1 

def wykres(w, d, x, line_alpha=0.2): #plots the dateset and the decision boundary
  xx = np.arange(-10, 10)
  a = -w[1]/w[2]
  b = -w[0]/w[2]
  yy = a*xx + b
  plt.plot(xx, yy, 'r', alpha=line_alpha)
  for i in range(len(x)):
    if d[i] == 1:
      plt.plot(x[i, 0], x[i, 1], 'go')
    else:
      plt.plot(x[i, 0], x[i, 1], 'ro')

  plt.axis([-10, 10, -10, 10])

def krok_uczenia(w, x, d, eta = 0.1): # a single learning step
  for i in range( len(x) ):
    xx = x[i]
    dd = d[i]
    s = xx[0] * w[1] + xx[1] * w[2] + 1 * w[0]
    if s > 0:
        y = 1
    else:
        y = 0
    w[0] = w[0] + eta*(dd-y)*1
    w[1] = w[1] + eta*(dd-y)*xx[0]
    w[2] = w[2] + eta*(dd-y)*xx[1]

def perceptron_lepszy(w, x, d, eta=0.1, max_iter=100000): # main training loop
  for ii in range(max_iter):
    wszystkie_ok = True
    print(f"Iteration number: {ii + 1}")
    for i in range(len(x)):
      xx = x[i]
      dd = d[i]
      s = xx[0] * w[1] + xx[1] * w[2] + 1 * w[0]
      if s > 0:
        y = 1
      else:
        y = 0

      print(f"x1: {xx[0]}, x2: {xx[1]}, d: {dd} -> y: {y}")

      if y != dd:
        wszystkie_ok = False
        krok_uczenia(w, np.array([xx]), np.array([dd]), eta)
    wykres(w, d, x)
    if wszystkie_ok:
      print(f"we got the proper result after {ii + 1} iterations.")
      print(f"{w}")
      return ii+1
      break
  if not wszystkie_ok:
      print(f"it didn't learn in {max_iter} itertaions.")

print("TEST 1") # testing the and logic gate
x1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
d1 = np.array([0, 0, 0, 1])
w1 = losuj_wagi(42)
print(w1)
wynik_1 = perceptron_lepszy(w1, x1, d1)
plt.show()

print("TEST 2") # testing the nand logic gate
x1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
d1 = np.array([0, 0, 0, 1])
w1 = losuj_wagi(42)
print(w1)
wynik_1 = perceptron_lepszy(w1, x1, d1)
x2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
d2 = np.array([1, 1, 1, 0])
w2 = losuj_wagi(42)
print(w2)
wynik_2 = perceptron_lepszy(w2, x2, d2)
print(f"first was done in {wynik_1} iterations, second in {wynik_2} iterations, so the difference is {wynik_2 - wynik_1} iterations")
plt.show()


print("TEST 3") # Test with a large, custom dataset
p = np.array([[3, 1], [0.3, 0.2], [5, 2], [8, 3], [10, 5],
              [3, 2], [4, 2], [7, 2], [2, 0],
              [1, 2], [6.5, 2.5], [4.5, 1], [5.2, 0.4],
              [3.2, 0.1], [4, 0.5], [6, 0.4], [6.4, 4.1], [6.8, 3.5], [5.2, 3.2],
              [1, 1], [1, 7.7], [2, 3], [5, 8], [1, 4], [2, 8], [2, 5], [0, 3], [2.3, 3.2],
              [5.3, 5.2], [2, 6.4], [2, 3.2], [2, 3.3], [2.1, 8.7],[9.3, 9.9],
              [1, 7.7], [5, 4.3], [2, 3.3], [4.5, 3.3], [3.4, 3.8], [6, 5.2], [6, 4.5]])
l = np.zeros(len(p))
ile = -1
for i in p:
  x = i[0]
  y = i[1]
  ile += 1
  if y > 0.5 * x + 1.1:
    plt.plot(x, y, 'o', color='gray')
    l[ile] = 1
  else:
    plt.plot(x, y, 'o', color='black')
x2 = np.linspace(0, 10, 200)
y2 = 0.5 * x2 + 1.1
plt.plot(x2, y2, color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
w1 = losuj_wagi(42)
perceptron_lepszy(w1, p, l)
plt.show()

print("Test 4") # test with pre-defined weights and new points
p = np.array([[3, 1], [0.3, 0.2], [5, 2], [8, 3], [10, 5],
              [3, 2], [4, 2], [7, 2], [2, 0],
              [1, 2], [6.5, 2.5], [4.5, 1], [5.2, 0.4],
              [3.2, 0.1], [4, 0.5], [6, 0.4], [6.4, 4.1], [6.8, 3.5], [5.2, 3.2],
              [1, 1], [1, 7.7], [2, 3], [5, 8], [1, 4], [2, 8], [2, 5], [0, 3], [2.3, 3.2],
              [5.3, 5.2], [2, 6.4], [2, 3.2], [2, 3.3], [2.1, 8.7],[9.3, 9.9],
              [1, 7.7], [5, 4.3], [2, 3.3], [4.5, 3.3], [3.4, 3.8], [6, 5.2], [6, 4.5]])
l = np.zeros(len(p))
ile = -1
for i in p:
  x = i[0]
  y = i[1]
  ile += 1
  if y > 0.5 * x + 1.1:
    plt.plot(x, y, 'o', color='gray')
    l[ile] = 1
  else:
    plt.plot(x, y, 'o', color='black')
x2 = np.linspace(0, 10, 200)
y2 = 0.5 * x2 + 1.1
plt.plot(x2, y2, color='green')
plt.xlabel('x')
plt.ylabel('y')
w2 = np.array([-0.45091976, -0.75857139, 1.13398788])
pnowe= np.array([[2,5], [0,-5], [9,3]])
l = np.zeros(len(pnowe))
ile = -1
for i in pnowe:
  x = i[0]
  y = i[1]
  ile += 1
  if y > 0.5 * x + 1.1:
    plt.plot(x, y, 'o', color='gray')
    l[ile] = 1
  else:
    plt.plot(x, y, 'o', color='black')
x2 = np.linspace(0, 10, 200)
y2 = 0.5 * x2 + 1.1
plt.plot(x2, y2, color='green')
plt.show()
perceptron_lepszy(w2, pnowe, l)
plt.show()



print("Test 5") #test different learning rates
p = np.array([[3, 1], [0.3, 0.2], [5, 2], [8, 3], [10, 5],
              [3, 2], [4, 2], [7, 2], [2, 0],
              [1, 2], [6.5, 2.5], [4.5, 1], [5.2, 0.4],
              [3.2, 0.1], [4, 0.5], [6, 0.4], [6.4, 4.1], [6.8, 3.5], [5.2, 3.2],
              [1, 1], [1, 7.7], [2, 3], [5, 8], [1, 4], [2, 8], [2, 5], [0, 3], [2.3, 3.2],
              [5.3, 5.2], [2, 6.4], [2, 3.2], [2, 3.3], [2.1, 8.7],[9.3, 9.9],
              [1, 7.7], [5, 4.3], [2, 3.3], [4.5, 3.3], [3.4, 3.8], [6, 5.2], [6, 4.5]])
l = np.zeros(len(p))
ile = -1
for i in p:
  x = i[0]
  y = i[1]
  ile += 1
  if y > 0.5 * x + 1.1:
    plt.plot(x, y, 'o', color='gray')
    l[ile] = 1
  else:
    plt.plot(x, y, 'o', color='black')
x2 = np.linspace(0, 10, 200)
y2 = 0.5 * x2 + 1.1
plt.plot(x2, y2, color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
eta_values = [0.01, 0.1, 0.5]
for eta in eta_values:
  w1 = losuj_wagi(42)
  perceptron_lepszy(w1, p, l, eta)
  plt.show()
  print(f"for eta = {eta}")