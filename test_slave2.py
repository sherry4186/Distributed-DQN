import pickle
from time import sleep
import os

share_folder = '//ALB0218/Users/xinzhu_ye/PythonScript/share/'
share_memory_folder = share_folder + 'memory/'
share_memory_file = share_memory_folder + 'memory2.txt'
share_model_folder = share_folder + 'DQNmodel/'
share_model_file = share_model_folder + 'DQNmodel.txt'
end_flag_folder = share_folder + 'end_flag/'
end_flag_file = end_flag_folder + 'end_flag2.txt'

memory = [[0, 0, 0, 0, 0]]


def server(share_memory_file, share_model_file):
    global memory
    pickle.dump(memory, open(share_memory_file, 'wb'))
    while os.path.exists(share_memory_file):
        sleep(10)
    load_model(share_model_file)


def generate_data(share_memory_file):
    global memory
    memory = [[i + 10 for i in memory[0]]]
    print('generate_data:', memory)


def load_model(inputfile):
    while 1:
        if os.path.exists(inputfile):
            print('load_model')
            break


for count in range(10):
    generate_data(share_memory_file)
    server(share_memory_file, share_model_file)

end_flag = True
pickle.dump(end_flag, open(end_flag_file, 'wb'))
