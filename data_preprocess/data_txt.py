import os

path = 'D:/Lai/counting_dataset/test/dotden/jhu++/'
target_path = 'D:/Lai/counting_dataset/test/dotden/jhu++/'
cls = 'train'

img_list = os.listdir(path + cls + '/')

with open(target_path + cls + '.txt', 'w') as f:

    num = len(img_list)
    for i in range(0, num, 3):
        # f.write('/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i+1] + ' ' + '/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i] + '\n')
        # f.write(target_path + cls + '/' + img_list[i + 1] + ' ' + target_path + cls + '/' + img_list[i] + '\n')
        f.write(path + cls + '/' + img_list[i + 1] + ' ' + path + cls + '/' + img_list[i+2] + '\n')

# folder_set = os.listdir(path)
# folder_set = folder_set[0:5]

# for folder in folder_set:
#     if folder == 'f5':
#         img_list = os.listdir(os.path.join(path,folder))
#         with open(target_path + 'test' + '.txt', 'w') as f:
#
#             num = len(img_list)
#             for i in range(0, num, 3):
#                 # f.write('/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i+1] + ' ' + '/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i] + '\n')
#                 # f.write(target_path + cls + '/' + img_list[i + 1] + ' ' + target_path + cls + '/' + img_list[i] + '\n')
#                 f.write(path + '/' + folder + '/' + img_list[i + 1] + ' ' + path + '/' + folder + '/' + img_list[i+2] + '\n')
#
#     else:
#         img_list = os.listdir(os.path.join(path, folder))
#         with open(target_path + 'train' + '.txt', 'a') as f:
#
#             num = len(img_list)
#             for i in range(0, num, 3):
#                 # f.write('/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i+1] + ' ' + '/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i] + '\n')
#                 # f.write(target_path + cls + '/' + img_list[i + 1] + ' ' + target_path + cls + '/' + img_list[i] + '\n')
#                 f.write(path + '/' + folder + '/' + img_list[i + 1] + ' ' + path + '/' + folder + '/' + img_list[i+2] + '\n')


