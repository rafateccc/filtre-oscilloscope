import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mpe


Ve=397.64e-3

Vs=[400,396,417,435,421,402,402,394,393,392,392,392,392,392,392,393,393,391,391,388,392,388,388,389,390,383,380,377,372,365,354,332,302]
Vs2=[297,279,259,237,213,187.5,161,135,111.3,89,69,39,19.5,5.7,3.9,2.5,1.6,1,0.55,0.275 ]

f1=np.logspace(1,4.27875360095,num=len(Vs))
f2=np.logspace(4.30102999566, 5.17609125906, num=len(Vs2))





Gdb=[]
for i in Vs:
    i=i*1e-3
    Gdb.append(20*np.log10(i/Ve))


for i in Vs2:
    i=i*1e-3
    Gdb.append(20*np.log10(i/Ve))

f=np.append(f1,f2)
Gmax=max(Gdb)
G_c=Gmax-3 #Gain à -3dB pour déterminer fc approximative
#Trouver fc approximative en interpolant entre les deux points le plus proches de G_c
fc_precise = np.interp(G_c, Gdb[::-1], f[::-1])
print("Fréquence de coupure approximative :", fc_precise, "kHz")
fc =  3e4 # KHz
Q=0.707 #FACTEUR DE QUALITÉ (On n'a pas cherché a optimiser ce paramètre au maximum, étant donné que graphiquement on voit que le modèle RLC n'est pas le meilleur)
fo=2e4 #KHz  (pareil ici, on n'a pas cherché à optimiser ce paramètre au maximum)
G0 = np.mean(Gdb[:10]) #Moyenne entre les dix premiers valeurs de Gdb pour avoir le comportement à basse f
# GBUTTERWORTH1_db=G0 -10*np.log10(1+(f/fc)**(1*2)) MÊME CAS QUE RC 1ER ORDRE
GBW2_db=G0 -10*np.log10(1+(f/fc)**(2*2)) #Gdb d'un filtre BW ORDRE 2
GBW3_db=G0 -10*np.log10(1+(f/fc)**(3*2))#Gdb d'un filtre BW ORDRE 3
GBW4_db=G0 -10*np.log10(1+(f/fc)**(4*2))#Gdb d'un filtre BW ORDRE 4
GBW5_db=G0 -10*np.log10(1+(f/fc)**(5*2))#Gdb d'un filtre BW ORDRE 5
GBW6_db=G0 -10*np.log10(1+(f/fc)**(6*2))#Gdb d'un filtre BW ORDRE 6
GLCR_db=G0-10*np.log10((1-(f/fo)**2)**2+(f/(Q*fo))**2) #GLCR d'unt filtre RLC 
GRC_db = G0 - 10*np.log10(1 + (f/fo)**2) #Gdb d'un filtre RC 1er ordre théorique


# Calcul d'erreur
Gdb = np.array(Gdb) #Convertir en array pour pouvoir calculer mpe
GBW2_db = np.array(GBW2_db)
GBW3_db = np.array(GBW3_db)
GBW4_db = np.array(GBW4_db)
GBW5_db = np.array(GBW5_db)
GBW6_db = np.array(GBW6_db)
GLCR_db=np.array(GLCR_db)
GRC_db=np.array(GRC_db)
#Revenir en échelle linéaire pour le calcul de mpe
G=10**(Gdb/20)
GBW2=10**(GBW2_db/20)
GBW3=10**(GBW3_db/20)
GBW4=10**(GBW4_db/20)
GBW5=10**(GBW5_db/20)
GBW6=10**(GBW6_db/20)
GLCR=10**(GLCR_db/20)
GRC=10**(GRC_db/20)




#Boucle pour estimer la fréquence de coupure idéale pour chaque modèle qui semble approprié 

#BW6
"""
#Variables de contrôle
control_mae=mae(Gdb,GBW6_db)
control_mpe=mpe(G,GBW6)
fc_control=0
for i in range(30):
    fc = 2e4+i*10**3  # KHz
    GBW6_db=G0 -10*np.log10(1+(f/fc)**(6*2))
    GBW6=10**(GBW6_db/20)
    err_moy_abs_BW6=mae(Gdb,GBW6_db)
    err_100_BW6=mpe(G,GBW6)
    if err_moy_abs_BW6<control_mae:
        control_mpe=err_100_BW6
        control_mae=err_moy_abs_BW6
        fc_control=fc
print(fc_control)
print(control_mae)
print(control_mpe*100,"%")
"""
#On trouve 43000.0 kHz ; 1.5314421850562367dB ; 22.447814428481696 %
GBW6_db=G0 -10*np.log10(1+(f/43000.0)**(6*2))#Gdb d'un filtre BW ORDRE 6

#BW5
"""
#Variables de contrôle
control_mae=mae(Gdb,GBW5_db)
control_mpe=mpe(G,GBW5)
fc_control=0
for i in range(30):
    fc = 2e4+i*10**3  # KHz
    GBW5_db=G0 -10*np.log10(1+(f/fc)**(5*2))
    GBW5=10**(GBW5_db/20)
    err_moy_abs_BW5=mae(Gdb,GBW5_db)
    err_100_BW5=mpe(G,GBW5)
    if err_moy_abs_BW5<control_mae:
        control_mpe=err_100_BW5
        control_mae=err_moy_abs_BW5
        fc_control=fc
print(fc_control)
print(control_mae)
print(control_mpe*100,"%")
"""
 # On trouve fc=36000.0 kHz ; MAE =1.2962766511118888dB ; MPE=16.217380785361495%
GBW5_db=G0 -10*np.log10(1+(f/36000.0)**(5*2))#Gdb d'un filtre BW ORDRE 5

#BW4
"""
#Variables de contrôle
control_mae=mae(Gdb,GBW4_db)
control_mpe=mpe(G,GBW4)
fc_control=0
for i in range(30):
    fc = 2e4+i*10**3  # KHz
    GBW4_db=G0 -10*np.log10(1+(f/fc)**(4*2))
    GBW4=10**(GBW4_db/20)
    err_moy_abs_BW4=mae(Gdb,GBW4_db)
    err_100_BW4=mpe(G,GBW4)
    if err_moy_abs_BW4<control_mae:
        control_mpe=err_100_BW4
        control_mae=err_moy_abs_BW4
        fc_control=fc
print(fc_control)
print(control_mae)
print(control_mpe*100,"%")
"""

# On trouve fc=28000.0 kHz ; 1.609681127880335dB; 16.247314636920528 %
GBW4_db=G0 -10*np.log10(1+(f/28000.0)**(4*2))
#ERREUR BW4
err_moy_abs_BW4=mae(Gdb,GBW4_db)
err_100_BW4=mpe(G,GBW4)
print("Erreur absolue moyenne BW4:",1.609681127880335,"dB")
print("Erreur moyenne absolue en pourcentage BW4:", 16.247314636920528,"%")

#ERREUR BW5
err_moy_abs_BW5=mae(Gdb,GBW5_db)
err_100_BW5=mpe(G,GBW5)
print("Erreur absolue moyenne BW5:",1.2962766511118888,"dB")
print("Erreur moyenne absolue en pourcentage BW5:", 16.217380785361495,"%")

#ERREUR BW6
err_moy_abs_BW6=mae(Gdb,GBW6_db)
err_100_BW6=mpe(G,GBW6)
print("Erreur absolue moyenne BW6:",1.5314421850562367,"dB")
print("Erreur moyenne absolue en pourcentage BW6:", 22.447814428481696,"%")

#ERREUR BW2
err_moy_abs_BW2=mae(Gdb,GBW2_db)
err_100_BW2=mpe(G,GBW2)
print("Erreur absolue moyenne BW2:",err_moy_abs_BW2,"dB")
print("Erreur moyenne absolue en pourcentage BW2:", err_100_BW2*100,"%")

#ERREUR BW3
err_moy_abs_BW3=mae(Gdb,GBW3_db)
err_100_BW3=mpe(G,GBW3)
print("Erreur absolue moyenne BW3:",err_moy_abs_BW3,"dB")
print("Erreur moyenne absolue en pourcentage BW3:", err_100_BW3*100,"%")

#ERREUR LCR
err_moy_abs_LCR=mae(Gdb,GLCR_db)
err_100_LCR=mpe(G,GLCR)
print("Erreur absolue moyenne RLC:",err_moy_abs_LCR,"dB")
print("Erreur moyenne absolue en pourcentage RLC : ", err_100_LCR*100,"%")

#ERREUR RC
err_moy_abs_RC=mae(Gdb,GRC_db)
err_100_RC=mpe(G,GRC)
print("Erreur absolue moyenne RC:",err_moy_abs_RC,"dB")
print("Erreur moyenne absolue en pourcentage RC:", err_100_RC*100,"%")
"""
#RC 
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f, GRC_db, linestyle='--', label='RC 1er ordre')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()

#BUTTERWORTH 2
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f,GBW2_db, linestyle='--', label='BW 2ème ordre')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()


#BUTTERWORTH 3
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f, GBW3_db, linestyle='--', label='BW 3ème ordre')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()


#RLC PASSE-BAS
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f, GLCR_db,linestyle='--', label='RLC Passe-Bas')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()

#BW ORDRE 4
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f, GBW4_db, linestyle='--', label='BW 4ème ordre')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()



#BW ORDRE 5
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f, GBW5_db, linestyle='--', label='BW 5ème ordre')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()



#BW ORDRE 6
plt.figure()
plt.semilogx(f, Gdb, marker='o', linestyle='-', label='Mesure')
plt.semilogx(f, GBW6_db, linestyle='--', label='BW 6ème ordre')
plt.xlabel('Fréquence (kHz)')
plt.ylabel('Gain en dB')
plt.title('Gain en fonction de la fréquence')
plt.grid(True, which='both')
#plt.ylim(-25, 5)  si on veut enlèver les derniers points qui sont pas fidèles
plt.legend()
"""

plt.figure(figsize=(8,5))

# Données expérimentales    
plt.semilogx(f, Gdb, 'o', markersize=5, label='Mesure')

# Modèles Butterworth
plt.semilogx(f, GBW4_db, '-', linewidth=2, label='Butterworth ordre 4')
plt.semilogx(f, GBW5_db, '--', linewidth=2, label='Butterworth ordre 5')
plt.semilogx(f, GBW6_db, '-.', linewidth=2, label='Butterworth ordre 6')

plt.xlabel('Fréquence (kHz)')   # 
plt.ylabel('Gain (dB)')
plt.title('Comparaison des filtres Butterworth')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.figure(figsize=(8,5))

# Données expérimentales 
plt.semilogx(f, Gdb, 'o', markersize=5, label='Mesure')

# Autres modèles
plt.semilogx(f, GRC_db, '--', linewidth=2, label='RC 1er ordre')
plt.semilogx(f, GLCR_db, '-.', linewidth=2, label='RLC passe-bas')
plt.semilogx(f, GBW2_db, ':', linewidth=2, label='Butterworth ordre 2')
plt.semilogx(f, GBW3_db, '-', linewidth=2, alpha=0.7, label='Butterworth ordre 3')

plt.xlabel('Fréquence (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Comparaison avec des modèles théoriques simples')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()

plt.show()
