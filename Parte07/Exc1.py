#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    print('Created a figure.')
    #plt.waitforbuttonpress()
    #plt.show

    #pts = plt.ginput(3)
   
    # ------------------------------------------
    # Execution
    # ------------------------------------------
    pts = {'xs': [], 'ys':[]}   #lista/dicionário
    while True:
        plt.plot(pts['xs'],pts['ys'],'rx', linewidth = 2, markersize = 12)

        pt = plt.ginput(1)

        if not pt:
            print('Terminated')
            break

        print('pt = ' + str(pt))

        pts['xs'].append(pt[0][0])
        pts['ys'].append(pt[0][1])


        print('pts = ' + str(pts))


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    #Escrever os pontos num ficheiro
    file = open('pts.pk1','wb')
    pickle.dump(pts,file)
    file.close


if __name__ == "__main__":
    main()
