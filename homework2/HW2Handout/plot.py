import matplotlib.pyplot as plt

index = 0
step = 5000
update = []
reward = []

file = open("test.txt", "r")
while True:
	line = file.readline()
	if line:
		if line[0:3] == "Eva":
			reward.append(float(line[27:]))
			update.append(step*index)
			index += 1
			# print float(line[27:])
	else:
		break

file.close()

plt.plot(update, reward, c='b', label='experience replay for MountainCar-v0')

plt.legend(loc='best')
plt.ylabel('average reward on 20 episodes')
plt.xlabel('update to Q-network')
plt.grid()
plt.show()
