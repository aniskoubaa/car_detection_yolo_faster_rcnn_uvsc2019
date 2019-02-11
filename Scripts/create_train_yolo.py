import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

def create_train_txt(read_path):
    text_file = open("trained.txt", "w") 
    #Modify to use glob to just pick jpg files from the 
    for filename in os.listdir(read_path):
    	text_file.write("%s/%s\n" %(read_path,filename))    
	
    text_file.close()



#cwd should be the write path
read_path = '/home/riotu/darknet/dataset/images'
write_path='/home/riotu/darknet/dataset'
create_train_txt(read_path)

