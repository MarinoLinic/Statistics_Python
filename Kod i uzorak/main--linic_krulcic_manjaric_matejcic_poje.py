import statistics as stat
import scipy as sp
import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.style.use('ggplot')

podaciUnsorted = pd.read_excel('place.xls', sheet_name='Stranica')
podaci = podaciUnsorted['PLACE_U_HRK'].tolist()

# 1. zadatak ======================================================
grupiranje = (max(podaci)-min(podaci)) / 10

zavrsne_vrijednosti_razreda=[]
temp = min(podaci) + grupiranje
zavrsne_vrijednosti_razreda.append(temp)
for i in range(1,10):
    temp2=zavrsne_vrijednosti_razreda[i-1] + grupiranje
    zavrsne_vrijednosti_razreda.append(temp2)

frekvencija=[]
numerator=0
for j in podaci:
	if j<zavrsne_vrijednosti_razreda[0]:
		numerator+=1
frekvencija.append(numerator)

for i in range(1,10):
	br=0
	for j in podaci:
		if zavrsne_vrijednosti_razreda[i-1] < j <= zavrsne_vrijednosti_razreda[i]:
			br=br+1
	frekvencija.append(br)

relFrekv=[]
for i in frekvencija:
	relFrekv.append(i/len(podaci))

#ispis tablice
tablica={'Grupa (razred)': list(range(1,11)), 'Iznos plaće u razredu': zavrsne_vrijednosti_razreda, 'Frekvencija': frekvencija, 'Relativna frekvencija': relFrekv}
with open('tablica.txt', 'w') as f:
  f.write(tabulate(tablica, headers='keys'))
print(tabulate(tablica, headers='keys'))
f.close()

#histogram frekvencija
fig = plt.figure(1, dpi=300)
plt.grid(axis='y', alpha=0.6)
hist = plt.hist(podaci, bins=10, color='#4da2c4', edgecolor='black')
plt.title('Histogram frekvencija plaća')
plt.xlabel('Iznos plaća')
plt.ylabel('Frekvencija iznosa plaća')

# LINIJE KOJE NISU POTREBNE NEK SE ZAKOMENTIRAJU OVISNO O ZADATKU
min_ylim, max_ylim = plt.ylim()
# aritm sredina
plt.axvline(stat.mean(podaci), color='#3f007d', linestyle='dashed', linewidth=2)
# geometrijska sredina
plt.axvline(stat.geometric_mean(podaci), color='#54278f', linestyle='dashed', linewidth=2)
# harmonijska sredina
plt.axvline(stat.harmonic_mean(podaci), color='#6a51a3', linestyle='dashed', linewidth=2)
# mod
plt.axvline(stat.mode(podaci), color='#111c70', linestyle='solid', linewidth=2)
# medijan
plt.axvline(stat.median(podaci), color='#0b406b', linestyle='solid', linewidth=2)
# 1. percentil
plt.axvline(np.percentile(podaci,1), color='#737373', linestyle='dotted', linewidth=2)
# 10. percentil
plt.axvline(np.percentile(podaci,10), color='#525252', linestyle='dotted', linewidth=2)
# 25. percentil
plt.axvline(np.percentile(podaci,25), color='#252525', linestyle='dotted', linewidth=2)
# 75. percentil
plt.axvline(np.percentile(podaci,75), color='#000000', linestyle='dotted', linewidth=2)

plt.legend(['Aritm. sredina', 'Geo. sredina', 'Harm. sredina', 'Mod', 'Medijan', '1. perc.', '10. perc.', '25. perc + Q1', '75. perc. + Q3'], loc='upper right', frameon=True, bbox_to_anchor=(1.1, 1), fontsize='xx-small')

plt.savefig('1.Histogram_frekvencija.png')


#histogram relativnih frekvencija
fig = plt.figure(2, dpi=300)
plt.grid(axis='y', alpha=0.6)
plt.hist(podaci, weights=np.zeros_like(podaci) + 1. / len(podaci), edgecolor='black', color='#4dc459')
plt.xlabel('Iznos plaća')
plt.ylabel('Relativna frekvencija iznosa plaća')
plt.title('Histogram relativnih frekvencija plaća')

min_ylim, max_ylim = plt.ylim()
# aritm sredina
plt.axvline(stat.mean(podaci), color='#3f007d', linestyle='dashed', linewidth=2)
# geometrijska sredina
plt.axvline(stat.geometric_mean(podaci), color='#54278f', linestyle='dashed', linewidth=2)
# harmonijska sredina
plt.axvline(stat.harmonic_mean(podaci), color='#6a51a3', linestyle='dashed', linewidth=2)

# mod
plt.axvline(stat.mode(podaci), color='#111c70', linestyle='solid', linewidth=2)
# medijan
plt.axvline(stat.median(podaci), color='#0b406b', linestyle='solid', linewidth=2)
# 1. percentil
plt.axvline(np.percentile(podaci,1), color='#737373', linestyle='dotted', linewidth=2)
# 10. percentil
plt.axvline(np.percentile(podaci,10), color='#525252', linestyle='dotted', linewidth=2)
# 25. percentil
plt.axvline(np.percentile(podaci,25), color='#252525', linestyle='dotted', linewidth=2)
# 75. percentil
plt.axvline(np.percentile(podaci,75), color='#000000', linestyle='dotted', linewidth=2)

plt.legend(['Aritm. sredina', 'Geo. sredina', 'Harm. sredina', 'Mod', 'Medijan', '1. perc.', '10. perc.', '25. perc + Q1', '75. perc. + Q3'], loc='upper right', frameon=True, bbox_to_anchor=(1.1, 1), fontsize='xx-small')

plt.savefig('1.Histogram_relativnih_frekvencija.png')
#plt.show()



# 2. zadatak ======================================================
kum = sp.stats.cumfreq(podaci, 10) #pretpostavka: ostavljamo u 10 razreda
x = np.add(kum.lowerlimit, np.linspace(0, kum.binsize*kum.cumcount.size, kum.cumcount.size))

ktablica={'Razred': x, 'Kumulativna frekvencija': kum.cumcount}
with open('kumulativne_vrijednosti.txt', 'w') as km:
  km.write(tabulate(ktablica, headers='keys'))
km.close

boja = plt.get_cmap('viridis')
fig = plt.figure(3, dpi=300)
plt.grid(axis='y', alpha=0.6)
plt.bar(x, kum.cumcount, width=1185, color=boja.colors, edgecolor="black")
plt.xlabel('Iznos plaće')
plt.ylabel('Kumulativna frekvencija plaća')
plt.title('Histogram kumulativnih frekvencija plaća')
plt.savefig('2.Histogram_kumulativnih_frekvencija.png')



# 3. zadatak ======================================================
print('\n')
print("Aritmetička sredina uzorka: {}".format(stat.mean(podaci)))
print("Geometrijska sredina uzorka: {}".format(stat.geometric_mean(podaci)))
print("Harmonijska sredina uzorka: {}".format(stat.harmonic_mean(podaci)))
print('\n')



# 4. zadatak ======================================================
print("Mod uzorka: {}".format(stat.mode(podaci)))
print("Medijan uzorka: {}".format(stat.median(podaci)))
print('\n') 




# 5. zadatak ======================================================
print("Najveci element u uzorku: {}".format(max(podaci)))
print("Najmanji element u uzorku: {}".format(min(podaci)))
print("Raspon uzorka: {}".format(max(podaci) - min(podaci)))
print('\n')



# 6. zadatak ======================================================
print("Prvi kvartil: {}".format(np.percentile(podaci,25))) 
print("Treci kvartil: {}".format(np.percentile(podaci,75)))
print('\n') 



# 7. zadatak ======================================================
print("1. percentil: {}".format(np.percentile(podaci,1))) 
print("10. percentil: {}".format(np.percentile(podaci,10))) 
print("25. percentil: {}".format(np.percentile(podaci,25))) 
print("75. percentil: {}".format(np.percentile(podaci,75)))
print('\n') 



# 8. zadatak ======================================================
suma=0
for i in range(0,len(podaci)):
	suma=suma+abs(podaci[i]-stat.mean(podaci))
print("Suma apsolutnih odstupanja: {}".format(suma))
print("Prosječno apsolutno odstupanje: {}".format(suma/len(podaci)))
print('\n')



# 9. zadatak ======================================================
print("Varijanca uzorka: {}".format(stat.pvariance(podaci)))
print("Standardna devijacija uzorka: {}".format(stat.pstdev(podaci))) 
print("Korigirana varijanca uzorka: {}".format(stat.variance(podaci))) 
print("Korigirana standardna devijacija uzorka: {}".format(stat.stdev(podaci)))
print('\n')



# 10. zadatak =====================================================
boksplot_dizajn = {'patch_artist': True,
            'boxprops': dict(facecolor='#7d1b46', color='black', alpha=0.7),
            'capprops': dict(color='black', linewidth=2),
            'medianprops': dict(color='#7d1b46', linewidth=2),
            'whiskerprops': dict(color='black', linewidth=2, alpha=0.6)}
fig = plt.figure(4, dpi=300)
plt.grid(axis='y', alpha=0.6)
plt.boxplot(podaci, **boksplot_dizajn, labels=[''])
plt.ylabel('Frekvencija iznosa plaća')
plt.title('Kutijasti dijagram plaća')
plt.savefig('10.Kutijasti_dijagram.png')



# 11. zadatak =====================================================
print("Koeficijent asimetrije uzorka: {}".format(sp.stats.skew(podaci,axis=0,bias=True))) 
print("Pearsonova mjera asimetrije S.k1 uzorka: {}".format(((stat.mean(podaci)-(stat.mode(podaci)))/(stat.pstdev(podaci))))) 
print("Pearsonova mjera asimetrije S.k2 uzorka: {}".format(3*((stat.mean(podaci)-(stat.median(podaci)))/(stat.pstdev(podaci))))) 
print("Bowleyeva mjera asimetrije uzorka: {}".format(((np.percentile(podaci,25)+np.percentile(podaci,75))-2*(stat.median(podaci))/(np.percentile(podaci,75)-np.percentile(podaci,25)))))
print('\n') 



# 12. zadatak =====================================================
print("Mjera zaobljenosti: {}".format(sp.stats.kurtosis(podaci)))
print('\n')



# 13. zadatak =====================================================
fig = plt.figure(5, dpi=300)
plt.grid(axis='y', alpha=0.6)
plt.plot(zavrsne_vrijednosti_razreda, frekvencija, color = '#8a323c', linewidth=4)
plt.xlabel("Iznos plaća")
plt.ylabel("Frekvencija iznosa plaća")
plt.title("Linearni grafikon za distribuciju podataka")
plt.savefig('13.Linearni_grafikon.png')



# INTERVALI POUZDANOSTI -------------------------------------------
# IP 1. zadatak ===================================================
print('Interval pouzdanosti 95%: ', st.t.interval(alpha=0.95, df=len(podaci)-1, loc=np.mean(podaci), scale=st.sem(podaci)))

# IP 2. zadatak ===================================================
print('Interval pouzdanosti 85%: ', st.t.interval(alpha=0.85, df=len(podaci)-1, loc=np.mean(podaci), scale=st.sem(podaci)))



# EKSTRA | Normalna distribucija za prvu stranicu
mu = stat.mean(podaci)
sigma = stat.pstdev(podaci)
varx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.style.use('dark_background')
fig = plt.figure(6, dpi=300)
plt.grid(axis='y', alpha=0.6)
plt.plot(varx, st.norm.pdf(varx, mu, sigma), color = '#d9e627', linewidth=7)
plt.savefig('thebellcurve.png')