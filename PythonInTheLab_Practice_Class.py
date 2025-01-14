import matplotlib.pyplot as plt
import numpy as np

class Person:
    def __init__(self):
        self=0
        #self.name = name
        #self.age = age
    
    def myfunc(self, title, save):
        #print('Hello my name is ' + self.name + ' and I am %s years old!' % (self.age))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.arange(0, 1, 0.001)
        y = np.sqrt(x)
        ax.plot(x, y)
        ax.set_title('%s' % title)