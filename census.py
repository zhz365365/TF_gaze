import matplotlib.pyplot as plt

def census_file(file_address):
    file = open(file_address, 'r')
    theta = []
    phi = []
    for line in file:
    	content = line.strip().split(' ')
    	if float(content[1]) > -0.4 and float(content[1]) < 0.5 and float(content[2]) < 0.5 and float(content[2]) > -0.4:
    		theta.append(float(content[1]))
    		phi.append(float(content[2]))
    	del content
    file.close()

    census_name = file_address.split('/')[-1].split('.')[0]
    plt.scatter(x=theta, y=phi, s=4)
    plt.axis([-1, 1, -1, 1])
    plt.xlabel('theta')
    plt.ylabel('phi')
    plt.title(census_name)
    plt.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.tight_layout()
    plt.savefig(census_name + '.jpg')
    #plt.savefig(census_name + '.eps')
    plt.close('all')
   
if __name__ == '__main__':
    census_file('./Left/Train_left.txt')
    census_file('./Left/Test_left.txt')
    census_file('./Right/Train_right.txt')
    census_file('./Right/Test_right.txt')