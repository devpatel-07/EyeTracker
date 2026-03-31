import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_vectors(Pa, Pb, Pi):
    ax.clear()

    # Vectors
    A = Pi - Pa
    B = Pi - Pb

    # Draw vectors
    ax.quiver(*Pa, *A, color='r', linewidth=2)
    ax.quiver(*Pb, *B, color='b', linewidth=2)

    # Draw points
    ax.scatter(*Pa, color='red', label='Pa')
    ax.scatter(*Pb, color='blue', label='Pb')
    ax.scatter(*Pi, color='green', label='Pi')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Fixed limits (prevents jumping)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    ax.legend()

    plt.draw()
    plt.pause(0.01)

while True:
    try:
        print("\nEnter Pa (x y z):")
        Pa = np.array(list(map(float, input().split())))

        print("Enter Pb (x y z):")
        Pb = np.array(list(map(float, input().split())))

        print("Enter Pi (x y z):")
        Pi = np.array(list(map(float, input().split())))

        plot_vectors(Pa, Pb, Pi)

    except Exception:
        print("Invalid input, try again.")