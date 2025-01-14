from time import sleep
from threading import Thread

def long_function():
    print('Starting...')
    sleep(5)
    print('Finishing...')

# Threads is like a very fancy secretary. In Activity Monitor VSCode has 39 threads, python has 101 threads, but CPU is not increasing that much.
print('Before')
threads = [Thread(target=long_function) for i in range(100)]  #  we have 100 threads and they are all working at the same time!!
for t in threads:
    t.start()
'''t = Thread(target=long_function)
t2 = Thread(target=long_function)
t.start()
t2.start()'''
print('After')