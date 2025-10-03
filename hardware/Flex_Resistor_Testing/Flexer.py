import serial
import pandas as pd
import matplotlib.pyplot as plt

# --- Setup Serial ---
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)

# --- Collect Data ---
data = []
num_samples = 100  # how many lines you want to read

print("Reading data...")

for _ in range(num_samples):
    line = ser.readline().decode('utf-8').strip()
    if not line:
        continue
    try:
        # Assume numbers are comma-separated, like "123,45,200,15,99"
        values = list(map(int, line.split(',')))
        if len(values) == 5:  # Ensure we only log valid lines
            data.append(values)
            print(values)
    except ValueError:
        # Skip malformed lines
        continue

ser.close()

# --- Convert to DataFrame ---
df = pd.DataFrame(data, columns=['Val1', 'Val2', 'Val3', 'Val4', 'Val5'])

# --- Save to CSV for Excel ---
df.to_csv("serial_data.csv", index=False)
print("Data saved to serial_data.csv")

# --- Plot with Matplotlib ---
plt.figure(figsize=(10,6))
for col in df.columns:
    plt.plot(df.index, df[col], label=col)

plt.title("Serial Data")
plt.xlabel("Sample Number")
plt.ylabel("Value (0-255)")
plt.legend()
plt.grid(True)
plt.show()
