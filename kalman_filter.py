#**************Importing Required Libraries*************
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt


#*************Declare Variables**************************
#Read Input File
measurements = pd.read_csv('obj_pose-laser-radar-synthetic-input.txt', header=None, delim_whitespace = True, skiprows=1)

# Manualy copy initial readings from first row of input file.
prv_time = 1477010443000000/1000000.0

x = np.array([
        [0.312242],
        [0.5803398],
        [0],
        [0]
        ])

#Initialize variables to store ground truth and RMSE values
ground_truth = np.zeros([4, 1])
rmse = np.zeros([4, 1])

#Initialize matrices P and A
P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1000, 0],
        [0, 0, 0, 1000]
        ])
A = np.array([
        [1.0, 0, 1.0, 0],
        [0, 1.0, 0, 1.0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
        ])
H = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0]
        ])
I = np.identity(4)
z_lidar = np.zeros([2, 1])
R = np.array([
        [0.0225, 0],
        [0, 0.0225]
        ])
noise_ax = 5
noise_ay = 5
Q = np.zeros([4, 4])

#**********************Define Functions*****************************
def predict():
    # Predict Step
    global x, P, Q
    x = np.matmul(A, x)
    At = np.transpose(A)
    P = np.add(np.matmul(A, np.matmul(P, At)), Q)

def update(z):
    global x, P    
    # Measurement update step
    Y = np.subtract(z_lidar, np.matmul(H, x))
    Ht = np.transpose(H)
    S = np.add(np.matmul(H, np.matmul(P, Ht)), R)
    K = np.matmul(P, Ht)
    Si = inv(S)
    K = np.matmul(K, Si)
    
    # New state
    x = np.add(x, np.matmul(K, Y))
    P = np.matmul(np.subtract(I ,np.matmul(K, H)), P)

def CalculateRMSE(estimations, ground_truth):
    rmse = np.zeros([4, 1])
    #if (sys.getsizeof(estimations) != sys.getsizeof(ground_truth) or sys.getsizeof(estimations) == 0):
    #    print ('Invalid estimation or ground_truth data')
    #    return rmse
    rmse[0][0] =  np.sqrt(((estimations[0][0] - ground_truth[0][0]) ** 2).mean())
    rmse[1][0] =  np.sqrt(((estimations[1][0] - ground_truth[1][0]) ** 2).mean())
    rmse[2][0] =  np.sqrt(((estimations[2][0] - ground_truth[2][0]) ** 2).mean())
    rmse[3][0] =  np.sqrt(((estimations[3][0] - ground_truth[3][0]) ** 2).mean())
    #print(rmse)
    return rmse

state_x = []
state_y = []
ground_truth_x = []
ground_truth_y = []
measure_x = []
measure_y = []
def get_X_and_Y(x):
    #print(x)
    state_x.append(x[0][0])
    state_y.append(x[1][0])

def get_measurement(new_measurement):
    #print(new_measurement)
    measure_x.append(new_measurement[1])
    measure_y.append(new_measurement[2])

def get_ground_truth(ground_truth):
    #print(ground_truth)
    ground_truth_x.append(ground_truth[0][0])
    ground_truth_y.append(ground_truth[1][0])

#**********************Iterate through main loop********************
#Begin iterating through sensor data
for i in range (len(measurements)):
    new_measurement = measurements.iloc[i, :].values
    if new_measurement[0] == 'L':
        #Calculate Timestamp and its power variables
        cur_time = new_measurement[3]/1000000.0
        dt = cur_time - prv_time
        #print(dt)
        prv_time = cur_time
        dt_2 = dt * dt
        dt_3 = dt_2 * dt
        dt_4 = dt_3 * dt
        #Updating matrix A with dt value
        A[0][2] = dt
        A[1][3] = dt
        #Updating Q matrix
        Q[0][0] = dt_4/4*noise_ax
        Q[0][2] = dt_3/2*noise_ax
        Q[1][1] = dt_4/4*noise_ay
        Q[1][3] = dt_3/2*noise_ay
        Q[2][0] = dt_3/2*noise_ax
        Q[2][2] = dt_2*noise_ax
        Q[3][1] = dt_3/2*noise_ay
        Q[3][3] = dt_2*noise_ay
        #Updating sensor readings
        z_lidar[0][0] = new_measurement[1]
        z_lidar[1][0] = new_measurement[2]
        #Collecting ground truths
        ground_truth[0] = new_measurement[4]
        ground_truth[1] = new_measurement[5]
        ground_truth[2] = new_measurement[6]
        ground_truth[3] = new_measurement[7]
        #Call Kalman Filter Predict and Update functions.
        predict()
        update(z_lidar)
        get_X_and_Y(x)
        get_measurement(new_measurement)
        
    #print('iteration', i, 'x: ', x)
    rmse = CalculateRMSE(x, ground_truth)
    get_ground_truth(ground_truth)

states = np.stack((state_x,state_y), axis = 1)
#print(states)

ground_truth_state = np.stack((ground_truth_x,ground_truth_y), axis = 1)
#print(ground_truth_state)

measured_state = np.stack((measure_x, measure_y), axis = 1)
#print(measured_state)

plt.plot(ground_truth_state[:,0], ground_truth_state[:,1])
plt.plot(measured_state[:,0], measured_state[:,1])
plt.plot(states[:,0], states[:,1])

plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['ground truth', 'measurements', 'KF'])

plt.show()