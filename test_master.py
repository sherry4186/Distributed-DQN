import pickle
import os
import glob
from time import sleep


share_folder = '//ALB0218/Users/xinzhu_ye/PythonScript/share/'
share_memory_folder = share_folder + 'memory' + os.sep
share_memory_file1 = share_memory_folder + 'memory1.txt'
share_memory_file2 = share_memory_folder + 'memory2.txt'
share_model_folder = share_folder + 'DQNmodel' + os.sep
share_model_file = share_model_folder + 'DQNmodel.txt'
share_model_file_bk = share_model_folder + 'DQNmodel.txt.bk'
end_flag_folder = share_folder + 'end_flag' + os.sep
end_flag_file1 = end_flag_folder + 'end_flag1.txt'
end_flag_file2 = end_flag_folder + 'end_flag2.txt'

memory = []


def load_data(files):
    global memory
    for file in files:
        tmp_memory = pickle.load(open(file, 'rb'))
        memory.extend(tmp_memory)
        os.remove(file)
        print('load_data:', file)


def save_model(share_model_file, share_model_file_bk):
    global memory
    if os.path.exists(share_model_file):
        if os.path.exists(share_model_file_bk):
            os.remove(share_model_file_bk)
        os.rename(share_model_file, share_model_file_bk)
    pickle.dump(memory, open(share_model_file, 'wb'))
    print('save_model')


end_flag_files = glob.glob(end_flag_folder + '*')
for file in end_flag_files:
    os.remove(file)
while not (end_flag_file1 in end_flag_files and end_flag_file2 in end_flag_files):
    share_memory_files = glob.glob(share_memory_folder + '*')
    sleep(10)
    if len(share_memory_files) > 0:
        load_data(share_memory_files)
        save_model(share_model_file, share_model_file_bk)
    end_flag_files = glob.glob(end_flag_folder + '*')

memory_tmp = pickle.load(open(share_model_file, 'rb'))
print(memory_tmp)
print('yes!')
