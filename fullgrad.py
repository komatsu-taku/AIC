import matplotlib.pyplot as plt
import numpy as np


def relu(x, b=1):
    return b-x if b-x > 0 else 0

def main():

    def ReLU(x, a=1, b=1):
        return a - np.maximum(0, b-x)
    
    x = np.array(range(-5, 5))
    y = ReLU(x, 1)

    plt.title('full-gradients')
    plt.plot(x, y)
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    main()

