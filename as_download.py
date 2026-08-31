'''
================================================ 
          DOWNLOAD_AUDIOSET REPOSITORY                     
================================================ 

repository name: download_audioset 
repository version: 1.0 
repository link: https://github.com/jim-schwoebel/download_audioset 
author: Jim Schwoebel 
author contact: js@neurolex.co 
description: downloads the raw audio files from AudioSet (released by Google). 
license category: opensource 
license: Apache 2.0 license 
organization name: NeuroLex Laboratories, Inc. 
location: Seattle, WA 
website: https://neurolex.ai 
release date: 2018-11-08 

This code (download_audioset) is hereby released under a Apache 2.0 license license. 

For more information, check out the license terms below. 

================================================ 
                SPECIAL NOTES                     
================================================ 

This script parses through the entire balanced audioset dataset and downloads
all the raw audio files. The files are arranged in folders according to their
representative classes.

Please ensure that you have roughly 35GB of free space on your computer before
downloading the files. Note that it may take up to 2 days to fully download 
all the files.

Enjoy! - :) 

-Jim 

================================================ 
                LICENSE TERMS                      
================================================ 

Copyright 2018 NeuroLex Laboratories, Inc. 
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

     http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

================================================ 
                SERVICE STATEMENT                    
================================================ 

If you are using the code written for a larger project, we are 
happy to consult with you and help you with deployment. Our team 
has >10 world experts in Kafka distributed architectures, microservices 
built on top of Node.js / Python / Docker, and applying machine learning to 
model speech and text data. 

We have helped a wide variety of enterprises - small businesses, 
researchers, enterprises, and/or independent developers. 

If you would like to work with us let us know @ develop@neurolex.co. 
'''

################################################################################
##                            IMPORT STATEMENTS                               ##
################################################################################

import pafy, os, shutil, time, ffmpy
import pandas as pd
import soundfile as sf 
from tqdm import tqdm

################################################################################
##                            HELPER FUNCTIONS                                ##
######################   ##########################################################

#function to clean labels 
def convertlabels(sortlist,labels,textlabels):
    clabels=list()
    label_dict=dict(zip(labels, textlabels))
    sortlist = sortlist.split(",")
    for i in range(len(sortlist)):
        #pull out converted label
        clabels.append(label_dict[sortlist[i]])
    return clabels 

def download_audio(link):
    listdir=os.listdir()
    try:
        # 使用python -m yt_dlp命令
        result = os.system(".venv\\Scripts\\python.exe -m yt_dlp -f 'bestaudio[ext=m4a]' '%s'"%(link))
        if result != 0:
            # 如果yt-dlp失败，尝试youtube-dl
            result = os.system("youtube-dl -f 'bestaudio[ext=m4a]' '%s'"%(link))
            if result != 0:
                print(f"无法下载: {link}")
                return None
    except Exception as e:
        print(f"下载出错: {e}")
        return None
    
    listdir2=os.listdir()
    filename=''
    for i in range(len(listdir2)):
        if listdir2[i] not in listdir and listdir2[i].endswith('.m4a'):
            filename=listdir2[i]
            break
    
    if not filename:
        print(f"未找到下载的文件: {link}")
        return None
        
    return filename

def is_valid_youtube_id(youtube_id):
    """检查YouTube ID是否有效"""
    # 检查是否为空或None
    if not youtube_id or youtube_id is None:
        return False
    
    # 检查是否以无效字符开头
    if youtube_id.startswith('--') or youtube_id.startswith('-0') or youtube_id.startswith('-1'):
        return False
    
    # 检查长度（YouTube ID通常是11个字符）
    if len(youtube_id) != 11:
        return False
    
    # 检查是否包含无效字符
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        if char in youtube_id:
            return False
    
    # 检查是否包含连续的特殊字符
    if '--' in youtube_id:
        return False
    
    return True

################################################################################
##                            MAIN SCRIPT                                     ##
################################################################################

defaultdir=os.getcwd()
os.chdir(defaultdir)

#load labels of the videos

#number, label, words
loadfile=pd.read_excel('labels.xlsx')

number=loadfile.iloc[:,0].tolist()
labels=loadfile.iloc[:,1].tolist()
textlabels=loadfile.iloc[:,2].tolist()
#remove spaces for folders 
for i in range(len(textlabels)):
    textlabels[i]=textlabels[i].replace(' ','')

#now load data for youtube
loadfile2=pd.read_excel('balanced_train_segments.xlsx')

# ylabels have to be cleaned to make a good list (CSV --> LIST) 
yid=loadfile2.iloc[:,0].tolist()[2:]
ystart=loadfile2.iloc[:,1].tolist()[2:]
yend=loadfile2.iloc[:,2].tolist()[2:]
ylabels=loadfile2.iloc[:,3].tolist()[2:]
 
#make folders
try:
    defaultdir2=os.path.abspath(os.getcwd())+'/audiosetdata/'
    if not os.path.exists(defaultdir2):
        os.makedirs(defaultdir2)
    os.chdir(defaultdir2)
except Exception as e:
    print(f"创建主目录失败: {e}")
    defaultdir2=os.path.abspath(os.getcwd())+'/audiosetdata/'
    os.makedirs(defaultdir2, exist_ok=True)
    os.chdir(defaultdir2)

for i in range(len(textlabels)):
    try:
        os.makedirs(textlabels[i], exist_ok=True)
    except Exception as e:
        print(f"创建目录 {textlabels[i]} 失败: {e}")
        pass
        
#iterate through entire CSV file, look for '--' if found, find index, delete section, then go to next index
slink='https://www.youtube.com/watch?v='

for i in tqdm(range(len(yid))):
    # 检查YouTube ID是否有效（不以'--'开头）
    if not is_valid_youtube_id(yid[i]):
        print(f'跳过无效的YouTube ID: {yid[i]}')
        continue
        
    link=slink+yid[i]
    start=int(float(ystart[i]))
    end=int(float(yend[i]))
    clabels=convertlabels(ylabels[i],labels,textlabels)
    print(clabels)
    
    # 初始化变量
    snippedfile = None
    lastdir = None
    
    for j in range(len(clabels)):
        
        #change to the right directory
        newdir=defaultdir2+clabels[j]+'/'
        
        # 确保目录存在
        if not os.path.exists(newdir):
            try:
                os.makedirs(newdir)
            except Exception as e:
                print(f"无法创建目录 {newdir}: {e}")
                continue
                
        os.chdir(newdir)
        
        if j ==0:
            
            #if it is the first download, pursue this path to download video 
            lastdir=os.getcwd()+'/'
    
            try:
                # use YouTube DL to download audio
                filename=download_audio(link)
                if filename is None:
                    print(f"跳过下载: {link}")
                    continue

                extension='.m4a'
                #get file extension and convert to .wav for processing later 
                new_filename = '%s_start_%s_end_%s%s'%(str(i),start,end,extension)
                try:
                    os.rename(filename, new_filename)
                    filename = new_filename
                except Exception as e:
                    print(f"重命名文件失败: {e}")
                    continue
                    
                if extension not in ['.wav']:
                    xindex=filename.find(extension)
                    filename_base=filename[0:xindex]
                    try:
                        ff=ffmpy.FFmpeg(
                            inputs={filename:None},
                            outputs={filename_base+'.wav':None}
                            )
                        ff.run()
                        os.remove(filename)
                        filename = filename_base+'.wav'
                    except Exception as e:
                        print(f"音频转换失败: {e}")
                        continue
                
                file=filename
                try:
                    data,samplerate=sf.read(file)
                    totalframes=len(data)
                    totalseconds=totalframes/samplerate
                    startsec=start
                    startframe=int(samplerate*startsec)
                    endsec=end
                    endframe=int(samplerate*endsec)
                    
                    # 确保索引在有效范围内
                    if startframe >= len(data):
                        startframe = 0
                    if endframe > len(data):
                        endframe = len(data)
                    if startframe >= endframe:
                        print(f"时间范围无效: {startframe} >= {endframe}")
                        continue
                        
                    print(f"处理音频: {startframe} -> {endframe}")
                    sf.write('snipped'+file, data[startframe:endframe], samplerate)
                    snippedfile='snipped'+file
                    os.remove(file)
                    
                except Exception as e:
                    print(f"音频处理失败: {e}")
                    snippedfile = None
                
            except Exception as e:
               print(f'下载或处理音频时出错: {e}')
               snippedfile = None

        else:
            #copy if already downloaded to proper labeled directory
            #this will eliminated repeated youtube calls to download
            if snippedfile and lastdir:
                print('copying file to %s'%(newdir+snippedfile))
                try:
                    shutil.copy(lastdir+snippedfile,newdir+snippedfile)
                except Exception as e:
                    print(f'复制文件时出错: {e}')
            else:
                print('无法复制文件：snippedfile或lastdir未定义')

    #sleep 5 seconds to prevent IP from getting banned 
    time.sleep(5) 


