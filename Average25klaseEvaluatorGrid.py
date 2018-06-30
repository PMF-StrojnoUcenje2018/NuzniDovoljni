
from __future__ import division, print_function, absolute_import
import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from tqdm import tqdm
import os
import json
import numpy as np
from random import shuffle
import cv2 
import shutil
import csv
import time
import random

THRESHOLD=0.9
topk=5
#raspakirati u direktoriju i potom postaviti MY_DIR kao putanju do samog projekta.
#MY_DIR = '/home/bosnmar/MachineLearning/Production/'
MY_DIR = 'C:/Users/User/Desktop/KODOVI_MOZGALO'

#kod slanja zakomentirati ovaj model. 
#MODELNAME = "6LCONVSS"
MODELNAME = "punocisci29-06_mriza_best9862"


#direktoriji koji su bitni za rad programa
SOURCEIMG = "C:/Users/User/Desktop/KODOVI_MOZGALO/MacLearnImg/TestImgs/"
SAVELOC = MY_DIR + "/MacLearnImg/ResIMG/"

LR = 0.001
IMGSIZE1 = 60
IMGSIZE2 = 40

#Get labels for our logos 
def getLabels():
    with open('imgLabels.json') as f:
        return json.load(f)


#Loada keyeve
def getMyKeys():
    with open('claskey.json') as f:
        return json.load(f)

#loada omjere za klase tj koliko je puta selective search predložio groundtruth/ukupni broj prijedloga
def getClassOmjer():
    with open('omjerclass.json') as f:
        return json.load(f)


#proces data from given picture into
def processTestData(img):
    testData = []
    
    for props in os.listdir(SAVELOC + img+ "/"):
        #print(cv2.imread(os.path.join(SAVELOC, props)))
        cvimg = cv2.imread(os.path.join(SAVELOC + img+ "/", props), 0)
        imgNum = props.split('.')[0]
        #print (np.array(cvimg))
        testData.append([np.array(cvimg), imgNum])
    return testData


if __name__ == "__main__":

    labels = getLabels()

    #print (labels.keys())

    #omjeri ispravnih region proposala iz faze selective search compare with croped logo. sluze za kasni
    #omjeri = getClassOmjer()
    #ovdje je slozeno po tipu "redniBroj": "ImeKlase" sluzi da kasnije otkrijemo kojoj klasi pripada najveca vrijednsot iz tezinskog vektora
    myKeys = getMyKeys()
    
    #ovdje cemo sloziti dictionary po "imeklase" : broj ponavljanja u nekoj slici kasnije za svaki region proposal racunamo kojoj klasi pripada.
    tempRes = {}
    #ovdje slazemo navedeni dictionary
    for key in myKeys.keys():
        tempRes.update({myKeys[key]: 0})

    #malo optimizacije
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)
    
    #preloadanje training seta iz fiknsog filea. sadrži malo manje od pola milijona slika

    print("\n\n")
    print("Starting with convnet. Fingers crossed\n")
    #network = tflearn.input_data(shape=[None, IMGSIZE1, IMGSIZE2, 1], data_augmentation=img_aug, name ='input')
    
    network = input_data(shape=[None, IMGSIZE1, IMGSIZE2, 1], name='input')
    network = conv_2d(network, 50, 5, activation='relu')
    network = max_pool_2d(network, 2)
    #Normalization purposed to speed up training process
    network = local_response_normalization(network)
    network = conv_2d(network, 50, 5, activation='relu')
    network = avg_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 100, 5, activation='relu')
    network = avg_pool_2d(network, 2)
    #network = local_response_normalization(network)
    
    network = fully_connected(network, 1000, activation='relu')
    network = fully_connected(network, 1000, activation='relu')
    network = fully_connected(network, 200, activation='relu')
    network = fully_connected(network, 25, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                        loss='categorical_crossentropy', name='targets')

    
    model = tflearn.DNN(network, tensorboard_dir='log' + MODELNAME)

    if os.path.exists("{}.meta".format(MODELNAME)):
        model.load(MODELNAME)
        print("Model Loaded")
    
    #model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), batch_size =128, shuffle = True, snapshot_step=1000, show_metric=True, run_id=MODELNAME)

    #model.save("DeepCNN15+3+9EpochLR0.005")
    #tu je zavrsila obrada podataka i neural network Loadamo spremljeni model. sada krecemo s loadanjem slika 0
    
    #testne slike bi trebale biti spremljene u odgovarajuci SOURCEIMG kako je navedeno gore u originalnom formatu.
    testImages =  [img for img in os.listdir(SOURCEIMG)]
    Height=400
    Width=220
    #testImages=[]
    #u results spremamo zavrne podatke. broj slike : klasa kojoj je model odlucio da pripada.
    for k in range(76,98,2):
        TreshMax=k
        results = {}
     
        #prolazim po slikama
        for img in tqdm(testImages): 
            
            index = img.split('.')[0]
            #print(index)
            intindex=int(index)
        
            try:
                im = cv2.imread(SOURCEIMG +img)
                im = cv2.resize(im, (Width, Height))
                ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                ss.setBaseImage(im)
        
                #ss.switchToSingleStrategy()
                ss.switchToSelectiveSearchFast()
                    # run selective search segmentation on input image
                    
                    #funkcija vrati array sa koordinatama rectangleova [x1 y1 w h] w->sirina h->visina dakle x1 y1 x1+w y1+h so koord od recta
                rects = ss.process()
                threshold_rects=[]
        
                    #cuva samo rectangle takve da je povrsina >= povrsinaslike/100
                for i,rect in enumerate(rects):
                    x, y, w, h = rect
                    
                    if (w>(2*h/3) and w*h<Height*Width/15 and w*h>Height*Width/250): 
                    #if (w*h >= Height*Width/100):
                        threshold_rects.append(rect)
                
                n_proposala=int(len(threshold_rects))
                max_ind=np.zeros(shape=(n_proposala, 2))
                sume = np.zeros(shape=(25, 2))
                average = np.zeros(shape=(25, 1))
                imOut = im.copy()
                for i,rect in enumerate(threshold_rects):
                    x ,y, w, h = rect
                        #funkciji bb_intersection_over_union moras dati gornji lijevi i donji desni pa to izvuci iz Logopoza 
                    img_prop = imOut[y:y+h, x:x+w]
                    proposal = cv2.resize(img_prop,(IMGSIZE1, IMGSIZE2))
                    proposal = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
        
                    #cv2.imwrite(SAVELOC + imgName + "/" + str(i) + ".jpg" , proposal)
                    #imgNum = props.split('.')[0]
                    #print (np.array(cvimg))
                    img_data=np.array(proposal)
                    orig = img_data
                    data = img_data.reshape(-1, IMGSIZE1, IMGSIZE2, 1)
                    model_out = model.predict(data)[0]
                    maximum=max(model_out)
                    myInd = str(np.where(model_out==maximum)[0][0])
                    max_ind[i][0]=maximum
                    max_ind[i][1]=myInd
                
                    #temporaryResults[myKeys[myInd]] = temporaryResults[myKeys[myInd]] + 1
                    
                
                for i in range(0, n_proposala):
                    sume[int(max_ind[i][1])][0] = max_ind[i][0]+sume[int(max_ind[i][1])][0]
                    sume[int(max_ind[i][1])][1] = sume[int(max_ind[i][1])][1]+ 1
                for i in range(0, 25):
                    if(sume[i][1] > 1):
                       average[i][0]=np.divide(sume[i][0], sume[i][1])   
                       
                
                ind = np.argpartition(average[:,0], -topk)[-topk:]
                ind = ind[average[ind,0].argsort()]
                #ind[np.argsort(average[ind][0])]
                #print(ind[0])
                
                print("\n")
                for n in range(0,topk):
                    print("slika " + str(intindex) + " klasa " + str(myKeys[str(ind[n])]) + " P-> " + str(average[ind[n]][0]) + " regioni -> " + str(sume[ind[n]][1]) + "\n")
                    
                maxi=0
                maxTotKoef=0
                for i in range (0,topk):
                    if sume[ind[i]][1]<=2:
                        Pkoef=0
                    else:
                        if average[ind[i]][0]>=0.99:
                            Pkoef=(average[ind[i]][0]*1000)-900
                        elif average[ind[i]][0]>=0.90:
                            Pkoef=(average[ind[i]][0]*100)-10
                        elif average[ind[i]][0]>=0.70:
                            Pkoef=(average[ind[i]][0]*100)-20
                        else:
                            Pkoef=0
                    
                    TotKoef=Pkoef+sume[ind[i]][1]
                    
                    if TotKoef>maxTotKoef:
                        maxTotKoef=TotKoef
                        maxi=i
                        
                if maxTotKoef>=TreshMax:
                    results.update({intindex : myKeys[str(ind[maxi])]})
                else:
                    results.update({intindex : "Other"})  
                     
                        
              
        
            except:
                print("U exceptu\n")
                results.update({intindex: "Other"})
    
            #temporary results je dictionary koji se na pocetku sastoji od parova kljuc : vrijednost. kljucevi su imena klasa, a pocetna vrijednost je 0
        #print(lista)
        
        #ovo prvo ispod dobro zapise u datoteku rezultate
        #imedatoteke= str(TreshMax)+'resultsAverage.txt'
        #TreshMax=90
        Name="results"+str(TreshMax)+"Average.txt"
    
        with open(  Name , 'w+') as csv_file:
            writer = csv.writer(csv_file)    
            for key in sorted(results):
                writer.writerow([results[key]])



    #print(results)

    #tu dodati funkciju koja zapisuje podatke u csv file. obavezno pripaziti da se rezultati ispisuju po redoslijedu 1,2,3.... do kraja. Ispisuje se samo ime klase, ne i redni broj (koliko sam shvatio na stranici za predaju rjesenja)
        


