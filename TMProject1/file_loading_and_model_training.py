import keras
import glob

filename_list = glob.glob("D:/Inne Projekty z Programowania/TMProject1/TMProject1/MFCC_files/*.txt")
#key=number value=list of mfcc
loaded_files={0:[],1: [],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for name in filename_list:
    dict={}
    file_with_mfcc=open(name,"r")
    y=name.split("_")
    y=int(y[2])
    #print(y)
    mfcc_string=file_with_mfcc.read()
    
    mfcc_string_list=mfcc_string.split("]")
    
    list_of_all_lists=[]
    for elements_string in mfcc_string_list:
        elements_string.replace("[","")
        elements_string.replace("\n","")
        
        elements_string=elements_string[1:len(elements_string)]
        elements_string_list=elements_string.split(" ")
        
        templist=[]
        for element in elements_string_list:
            if element=="":
                continue
            else:
                element=float(element)
                templist.append(element)            
            if len(templist)==13:
                list_of_all_lists.append(templist)
    file_with_mfcc.close()
    loaded_files[y].append(list_of_all_lists)
    
print('\n')
#print(len(loaded_files[4]))
        

