import matplotlib.pyplot as plt
import numpy as np
from PythonInTheLab_Practice_Class import Person

def main():
    p1 = Person()
    p2 = Person()
    p1.myfunc(title='yolo', save='hiya')
    p2.myfunc(title='yolo2', save='hiya2')
if __name__ == '__main__':
    main()

plt.show()