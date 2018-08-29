import matplotlib.pyplot as plt

def test(classes, filename):
	kernel = 180 / classes
	file_input = open(filename, 'r')
	mean_x = 0
	mean_y = 0
	count = 0
	for line in file_input:
		content = line.strip().split(' ')
		x = (float(content[1]) + 1) * 90
		y = (float(content[2]) + 1) * 90
		x_ = int(x / kernel)
		y_ = int(y / kernel)

		x_left = x_ * kernel
		x_right = (x_ + 1) * kernel
		if abs(x_left - x) < abs(x_right - x):
			x_best = x_left
		else:
			x_best = x_right

		y_left = y_ * kernel
		y_right = (y_ + 1) * kernel
		if abs(y_left - y) < abs(y_right - y):
			y_best = y_left
		else:
			y_best = y_right

		mean_x += float(abs(x - x_best))
		mean_y += float(abs(y - y_best))
		count += 1

		#print('%.3f %.3f %d %d' % (x, y, x_best, y_best), file=file_output)
		del content

	mean_x /= float(count)
	mean_y /= float(count)

	file_input.close()
	return mean_x, mean_y

def draw(filename):
	file_output = open(filename.split('.')[0] + '_check.txt','w')
	classes_list = []
	x_list = []
	y_list = []
	classes = 6
	while classes < 361:
		x, y = test(classes, filename)
		print('when the classes = %d, the min mean error in vertical direction is %.3f, the min mean error in horizontal is %.3f'%(classes, x, y), file=file_output)
		classes_list.append(classes)
		x_list.append(x)
		y_list.append(y)
		classes += 1
	file_output.close()

	plt.subplot(211)
	plt.plot(classes_list, x_list)
	plt.xlabel('classes')
	plt.ylabel('degrees')
	plt.title('minimum mean degree error in vertical direction')
	plt.axis([0, 400, 0, 8])

	plt.subplot(212)
	plt.plot(classes_list, y_list)
	plt.xlabel('classes')
	plt.ylabel('degrees')
	plt.title('minimum mean degree error in horizontal direction')
	plt.axis([0, 400, 0, 8])

	plt.tight_layout()
	plt.savefig(filename.split('.')[0] + '_minimum_mean_error.eps')
	plt.show()

if __name__ == '__main__':
	draw('Test_left_MPIIGaze.txt')
	draw('Test_right_MPIIGaze.txt')
