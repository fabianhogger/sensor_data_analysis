import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt
class Converter():
    def quaternion_to_euler_pure(self,w,i,j,k):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw)
        
        Parameters:
        quaternion : array-like
            A 4 element array representing the quaternion (w, x, y, z)
        
        Returns:
        euler_angles : tuple
            A tuple of 3 elements representing the Euler angles (roll, pitch, yaw) in degrees
        """
        w, x, y, z = w,i,j,k
        
        # Compute roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Compute pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arctan2(sinp, np.sqrt(1 - sinp * sinp))

        # Compute yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Convert radians to degrees
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

        return [roll, pitch, yaw]
    def quaternion_to_euler(self,w,i,j,k):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw)
        
        Parameters:
        quaternion : array-like
            A 4 element array representing the quaternion (w, x, y, z)
        
        Returns:
        euler_angles : tuple
            A tuple of 3 elements representing the Euler angles (roll, pitch, yaw)
        """
        # Create a rotation object from the quaternion
        r = R.from_quat([w,i,j,k])
        # Convert to Euler angles
        euler_angles = r.as_euler('xyz', degrees=True)

        return euler_angles

if __name__=="__main__":
    df=pd.read_csv("C:/Users/draiz/Documents/python_projects/imu/test22.txt",sep='\t')
    print(df.head())
    converter=Converter()
    df['euler']=df.apply(lambda row:converter.quaternion_to_euler_pure(row['q.w'],row['q.i'],row['q.j'],row['q.k']),axis=1 )
    print(df.head())
    df['Timestamp']=df.apply(lambda row:row['Timestamp']/1000,axis=1)
    df['roll']=df.apply(lambda row:row['euler'][0],axis=1)
    df['pitches']=df.apply(lambda row:row['euler'][1],axis=1)
    df['yaws']=df.apply(lambda row:row['euler'][2],axis=1)
    print(df.head())

    df['monotonic_col'] = range(1, len(df) + 1)
    new=df[['roll','pitches','yaws']]
    new.to_csv("C:/Users/draiz/Documents/python_projects/imu/file2result.csv")

    # Plot the results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(df['Timestamp'],df['roll'] , 'b', label='Roll')
    plt.xlabel('Time [s]')
    plt.ylabel('Roll [°]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(df['Timestamp'], df['pitches'], 'g', label='Pitch')
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch [°]')
    plt.yticks(np.arange(df['pitches'].min(), df['pitches'].max()+1, 14))
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(df['Timestamp'], df['yaws'], 'r', label='Yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw [°]')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
