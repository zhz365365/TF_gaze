import operator
import matplotlib.pyplot as plt

def clean(N, Head_Pose, processname):
    filename = '/data/zengdifei/TF_gaze/backup/MPIIGaze/N_' + N + Head_Pose + '/test_result' + processname
    file_input = open(filename + '.txt', 'r')

    Sumcount = {}
    Totalcount = {}
    Averagecount = {}

    for line in file_input:
        content = line.strip().split(' ')
        steps = int(content[2])
        error_mean = float(content[5])

        if steps not in Sumcount.keys(): Sumcount[steps] = 0
        if steps not in Totalcount.keys(): Totalcount[steps] = 0

        Sumcount[steps] += error_mean
        Totalcount[steps] += 1

        del content

    for steps in Sumcount.keys():
        Averagecount[steps] = Sumcount[steps] / float(Totalcount[steps])

    sortedAveragecount = sorted(Averagecount.items(), key=operator.itemgetter(0))
    
    file_input.close()

    file_output = open(filename + '_clean.txt', 'w')
    for i in range(len(sortedAveragecount)):
        print(sortedAveragecount[i][0], ' ', sortedAveragecount[i][1], file = file_output)

    file_output.close()

    Smooth_size = 20

    class drawpic():
        pass

    pic = drawpic()
    pic.steps_list = []
    pic.error_list = []

    file_output = open(filename + '_clean_smooth.txt', 'w')
    min_error = 20
    for i in range(len(sortedAveragecount)):
        j = max(0, i - Smooth_size + 1)
        ans = 0; tot = 0
        while j <= i:
            ans += sortedAveragecount[j][1]
            tot += 1
            j += 1
        ans /= float(tot)
        print(sortedAveragecount[i][0], ' ', ans, file=file_output)
        if sortedAveragecount[i][1] < min_error:
            min_error = sortedAveragecount[i][1]
        pic.steps_list.append(sortedAveragecount[i][0])
        pic.error_list.append(ans)
    print('min_error', ' ', min_error, file=file_output)
    file_output.close()

    return pic

def draw_one_plt(pic, picname):

    plt.plot(pic.steps_list, pic.error_list)
    plt.xlabel('steps')
    plt.ylabel('error_mean')
    plt.title('mean degree error with steps')
    plt.axis([0, 150000, 2, 30])
    plt.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.tight_layout()
    plt.savefig(picname + '_MPIIGaze_smooth.eps')
    plt.close('all')

def draw_two_plt(pic_with_head_pose, pic_without_head_pose, picname):

    plt.plot(pic_with_head_pose.steps_list, pic_with_head_pose.error_list, 'r', label='with_head_pose')
    plt.plot(pic_without_head_pose.steps_list, pic_without_head_pose.error_list, 'b', label='without_head_pose')
    plt.legend(ncol=1)
    plt.xlabel('steps')
    plt.ylabel('error_mean')
    plt.title('mean degree error with steps in different poses')
    plt.axis([0, 100000, 2, 30])
    plt.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.tight_layout()
    plt.savefig(picname + '_smooth.eps')
    plt.close('all')

def draw_five_plt(pic_18, pic_60, pic_90, pic_180, pic_360, picname):
    plt.plot(pic_18.steps_list, pic_18.error_list, label='N = 18')
    plt.plot(pic_60.steps_list, pic_60.error_list, label='N = 60')
    plt.plot(pic_90.steps_list, pic_90.error_list, label='N = 90')
    plt.plot(pic_180.steps_list, pic_180.error_list, label='N = 180')
    plt.plot(pic_360.steps_list, pic_360.error_list, label='N = 360')
    plt.legend(ncol=1)
    plt.xlabel('steps')
    plt.ylabel('error_mean')
    plt.title('mean degree error with steps in different divide plans')
    plt.axis([0, 100000, 2, 30])
    plt.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.tight_layout()
    plt.savefig(picname + '_smooth.eps')
    plt.close('all')

def draw_four_direction(N, Head_Pose):

    print('Start the process of N_' + N + Head_Pose + '!')
    left_x_pic = clean(N, Head_Pose, '_left_x')
    left_y_pic = clean(N, Head_Pose, '_left_y')
    right_x_pic = clean(N, Head_Pose, '_right_x')
    right_y_pic = clean(N, Head_Pose, '_right_y')
    print('Draw the picture of N_' + N + Head_Pose + '!') 
    draw_one_plt(left_x_pic, '/data/zengdifei/TF_gaze/picture/N_' + N + Head_Pose + '_left_x')
    draw_one_plt(left_y_pic, '/data/zengdifei/TF_gaze/picture/N_' + N + Head_Pose + '_left_y')
    draw_one_plt(right_x_pic, '/data/zengdifei/TF_gaze/picture/N_' + N + Head_Pose + '_right_x')
    draw_one_plt(right_y_pic, '/data/zengdifei/TF_gaze/picture/N_' + N + Head_Pose + '_right_y')
    print('Finish the process of N_' + N + Head_Pose + '!')

def draw_two_pose(N):
    
    print('Start the process of N_' + N + '!')
    left_x_pic_with_head_pose = clean(N, '_with_head_pose', '_left_x')
    left_x_pic_without_head_pose = clean(N, '_without_head_pose', '_left_x')
    left_y_pic_with_head_pose = clean(N, '_with_head_pose', '_left_y')
    left_y_pic_without_head_pose = clean(N, '_without_head_pose', '_left_y')
    right_x_pic_with_head_pose = clean(N, '_with_head_pose', '_right_x')
    right_x_pic_without_head_pose = clean(N, '_without_head_pose', '_right_x')
    right_y_pic_with_head_pose = clean(N, '_with_head_pose', '_right_y')
    right_y_pic_without_head_pose = clean(N, '_without_head_pose', '_right_y')
    print('Draw the picture of N_' + N + '! ')
    draw_two_plt(left_x_pic_with_head_pose, left_x_pic_without_head_pose, '/data/zengdifei/TF_gaze/picture/N_' + N + '_left_x')
    draw_two_plt(left_y_pic_with_head_pose, left_y_pic_without_head_pose, '/data/zengdifei/TF_gaze/picture/N_' + N + '_left_y')
    draw_two_plt(right_x_pic_with_head_pose, right_x_pic_without_head_pose, '/data/zengdifei/TF_gaze/picture/N_' + N + '_right_x')
    draw_two_plt(right_y_pic_with_head_pose, right_y_pic_without_head_pose, '/data/zengdifei/TF_gaze/picture/N_' + N + '_right_y')
    print('Finish the process of N_' + N + '!')

def draw_five_divide(Head_Pose, processname):
    print('Start the process of Pose' + Head_Pose + '_direction' + processname + '!')
    N_18 = clean('18', Head_Pose, processname)
    N_60 = clean('60', Head_Pose, processname)
    N_90 = clean('90', Head_Pose, processname)
    N_180 = clean('180', Head_Pose, processname)
    N_360 = clean('360', Head_Pose, processname)
    print('Draw the picture of Pose' + Head_Pose + '_direction' + processname + '!')
    draw_five_plt(N_18, N_60, N_90, N_180, N_360, '/data/zengdifei/TF_gaze/picture/Pose' + Head_Pose + '_direction' + processname)
    print('Finish the process of Pose' + Head_Pose + '_direction' + processname + '!')

def frontal_face(N, left_theta_is, left_phi_is, right_theta_is, right_phi_is):
    print('Draw the picture of frontal_face for ' + N + '!')
    if(left_theta_is == 1):
        left_theta = clean(N, '_with_frontal_face', '_left_theta')
        draw_one_plt(left_theta, '/data/zengdifei/TF_gaze/picture/frontal_face_' + N + '_left_theta')
    if(left_phi_is == 1):
        left_phi = clean(N, '_with_frontal_face', '_left_phi')
        draw_one_plt(left_phi, '/data/zengdifei/TF_gaze/picture/frontal_face_' + N + '_left_phi')
    if(right_theta_is == 1):
        right_theta = clean(N, '_with_frontal_face', '_right_theta')
        draw_one_plt(right_theta, '/data/zengdifei/TF_gaze/picture/frontal_face_' + N + '_right_theta')
    if(right_phi_is == 1):
        right_phi = clean(N, '_with_frontal_face', '_right_phi')
        draw_one_plt(right_phi, '/data/zengdifei/TF_gaze/picture/frontal_face_' + N + '_right_phi')
    print('Finish the picture of frontal_face for ' + N + '!')

if __name__ == '__main__':

    """
    draw_four_direction('18', '_with_head_pose')
    draw_four_direction('60', '_with_head_pose')
    draw_four_direction('90', '_with_head_pose')
    draw_four_direction('180', '_with_head_pose')
    draw_four_direction('360', '_with_head_pose')
    draw_four_direction('18', '_without_head_pose')
    draw_four_direction('60', '_without_head_pose')
    draw_four_direction('90', '_without_head_pose')
    draw_four_direction('180', '_without_head_pose')
    draw_four_direction('360', '_without_head_pose')
    
    draw_two_pose('18')
    draw_two_pose('60')
    draw_two_pose('90')
    draw_two_pose('180')
    draw_two_pose('360')

    draw_five_divide('_with_head_pose', '_left_x')
    draw_five_divide('_with_head_pose', '_left_y')
    draw_five_divide('_with_head_pose', '_right_x')
    draw_five_divide('_with_head_pose', '_right_y')
    draw_five_divide('_without_head_pose', '_left_x')
    draw_five_divide('_without_head_pose', '_left_y')
    draw_five_divide('_without_head_pose', '_right_x')
    draw_five_divide('_without_head_pose', '_right_y')
    """

    """
    frontal_face('180', 1, 1, 1, 1)
    frontal_face('120', 0, 1, 1, 1)
    frontal_face('90', 1, 1, 1, 1)
    frontal_face('60', 1, 1, 1, 1)
    frontal_face('30', 0, 1, 1, 0)
    frontal_face('18', 1, 0, 0, 1)
    """

    frontal_face('90', 1, 1, 1, 1)
    frontal_face('180', 1, 1, 1, 1)
    #frontal_face('360', 1, 1, 1, 1)
