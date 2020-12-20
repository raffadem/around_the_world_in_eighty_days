# around_the_world_in_eighty_days

P8: Il giro del mondo in 80 giorni
Si consideri l'insieme dei dati che descrivono alcune delle principali città del mondo (link). Si supponga che sia sempre possibile viaggiare da ogni città fino alle 3 città più vicine e che tale viaggio richieda 2 ore per la città più vicina, 4 ore per la seconda città più vicina e 8 ore per la terza più vicina. Inoltre il viaggio richiede 2 ore aggiuntive se la città di destinazione è in un'altra nazione rispetto alla città di partenza e altre 2 ore aggiuntive se la città di destinazione ha più di 200 000 abitanti.

Partendo la Londra e viaggiando sempre verso Est, è possible compiere il giro del mondo tornando a Londra in 80 giorni? Quanto tempo si impiega al minimo?

### step1: import e analisi dati
1. importare i dati 
2. modifica dati population come flag (gt 200K)
3. decodificare le colonne latitudine e longitudine in ascissa e ordinata(floating point)
4. ordinare per longitudine crescente

### step2: creazione griglia
1. creare griglia(dimensione variabile in base al numero(soglia, inferiore >=3 e superiore<=10(?)) città?)

### step3: calcolo peso
1. calcolare misura euclidea delle n città nel perimetro
2. date le 3 più vicine e poi assegnare gli ulteriori pesi(popolazione(flag già calcolato) + if su country)
3. prendere il peso minore, in caso di parità di peso prendere la città più lontana(longitudine)
4. reiterato fino a quando non arrivi a londra(di nuovo)
5. while(reitera fino a quando non c'è londra nel tuo quadrato(tra le 3 più vicine), in quel caso prendi londra)
(attenzione che londra ci deve stare, pensare ad una striscia di terra)

### stepn: visualizzazione con geopandas
1. visualizzare le città e il cammino con geopandas