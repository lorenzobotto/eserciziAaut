# Esercizi AAUT
Esercizi di Machine Learning: Decision Trees, KNN, Clustering e Probabilistic Models - Corso di Apprendimento Automatico - Unito 2023.

## Come visualizzare i risultati
Per visualizzare i risultati, è possibile scaricare il repository e aprire i file .ipynb con Jupyter Notebook.

Oppure è possibile visualizzare i risultati direttamente su GitHub, cliccando sui file .ipynb.

## Descrizione Esercizi
Durante il corso di Apprendimento Automatico, sono stati proposti degli esercizi da svolgere e da portare all'esame per poter comprendere meglio i concetti visti a lezione su alcuni dei modelli principali di ML. Questi esercizi sono stati svolti in Python (Jupyter Notebook), utilizzando le librerie scikit-learn e numpy.

Per ogni esercizio c'è una parte introduttiva già data dal docente, e una parte in cui noi studenti dobbiamo implementare varie richieste sul modello di ML in questione.

## Esercizi
### Esercizio 1 - Decision Trees
Implementazione dei modelli Decision Tree.

- [Nella parte introduttiva]: sono state spiegate le basi del modello Decision Tree, e sono state date delle istruzioni per implementare il modello Decision Tree. Inoltre, viene valutato il modello tramite la cross-validation ed è stato visualizzato il modello tramite la libreria graphviz.
- [Nella parte da implementare]: 
    1. viene applicato un sovracampionamento (artificial inflation) a due classi nel training set con un determinato fattore: 10, in modo da visualizzare come si comporta l'albero. Si apprende l'albero di decisione in queste condizioni;
    2. vengono appresi gli alberi cercando di evitare l'overfitting (migliorando l'errore sul test set) facendo 'tuning' degli iper-parametri: il minimo numero dei campioni per foglia, la massima profondità dell'albero, i parametri di minomo decremento dell'impurezza, massimo numero dei nodi foglia, ecc;
    3. viene costruita la matrice di confusione dell'albero creato sul test set e viene visualizzata;
    4. vengono costruite le curve ROC (o curve nello spazio di coverage) e si mostrano per ciascuna classe come si comportano, quali sono i valori di accuratezza sotto la curva e quali sono gli errori effettuati.


### Esercizio 2 - KNN
Implementazione del modello KNN.

- [Nella parte introduttiva]: sono state spiegate le basi del modello KNN, e sono state date delle istruzioni per implementare il modello KNN. Inoltre, vengono effettuate le seguenti operazioni:
    - il grafico dell'errore di classificazione che k-nn commette su Iris dataset con un crescente valore di k;
    - trova il valore ottimale di k;
    - analizza l'effetto della pesatura dei voti dei vicini con la distanza dei vicini;
    - il grafico dell'errore di classificazione con la pesatura dei voti in base alle distanze dei vicini;
    - trova il valore ottimale di k.

- [Nella parte da implementare]:
    1. mostrare lo scatter plot (in 2D, scegliendo 2 delle 4 features) dei dati di Iris, con un colore determinato dal modello KNN (colore rosso per Setosa, blu per Versicolor, verde per Virginica) e scelta delle 2 features migliori per la classificazione;
    2. plot del ROC Plot per il modello KNN, per ogni classe;
    3. visualizzare il ROC plot del migliore albero di decisione che abbiamo addestrato nell'esercizio n.1;
    4. confrontare gli alberi decisione e k-nn sullo spazio ROC: capire per quali valori di (TPR,FPR) k-nn è migliore rispetto agli alberi di decisione?
    5. eseguire k-nn ma ora usare come funzione di distanza una funzione:
distance(x,y)= 1- k(x,y). Dove k(x,y) è un Kernel Gaussian-like k(x,y) (per k(x,y) usare la Radial Basis Function con il parametro Gamma  = 1/sigma^2) che controlla la sua ampiezza. Il parametro gamma deve essere aggiustato (tuned) al valore ottimale, a secondo dell'accuratezza del k-nn, con k=7.

### Esercizio 3 - Clustering
Implementazione dei principali modelli di Clustering (DBScan e K-Means).

- [Nella parte introduttiva]: sono state spiegate le basi dei modelli di Clustering, e sono state date delle istruzioni per implementare i modelli di Clustering e plottarli.

- [Nella parte da implementare]:
    1. viene applicato K-Means sul dataset 2 e viene plottato il risultato;
    2. viene applicato K-Means sul dataset 3 e viene plottato il risultato;
    3. viene effettuato una Silhouette Analysis per determinare il numero ottimale di cluster per il dataset 1;
    4. viene effettuato una Silhouette Analysis per determinare il numero ottimale di cluster per il dataset 2;
    5. Si rende più piccolo il dataset 3, in quanto la Silhouette Analysis sarebbe stata troppo lenta, estraendo il 30% dei dati in modo casuale;
    6. viene effettuato una Silhouette Analysis per determinare il numero ottimale di cluster per il dataset 3;
    7. vengono effettuati dei plot per visualizzare i valori di Silhouette per ogni cluster e si visualizzano i dati clusterizzati, per il dataset 1.
    Questo viene effettuato per diversi valori di k, per vedere come cambiano i valori di Silhouette e come cambiano i cluster;
    8. vengono effettuati dei plot per visualizzare i valori di Silhouette per ogni cluster e si visualizzano i dati clusterizzati, per il dataset 2.
    Questo viene effettuato per diversi valori di k, per vedere come cambiano i valori di Silhouette e come cambiano i cluster;
    9. vengono effettuati dei plot per visualizzare i valori di Silhouette per ogni cluster e si visualizzano i dati clusterizzati, per il dataset 3.
    Questo viene effettuato per diversi valori di k, per vedere come cambiano i valori di Silhouette e come cambiano i cluster;
    10. viene applicato DBScan, con parametri casuali, sul dataset 2 e viene plottato il risultato;
    11. viene applicato DBScan, con parametri casuali, sul dataset 3 e viene plottato il risultato;
    12. viene utilizzato Nearest Neighbors per trovare il valore ottimale di eps per il dataset 3 (prima si fa con k=10 per far vedere che viene sbagliato e poi con k=50) e poi per il dataset 2:
        - vengono calcolate le distanze dei k vicini per ogni punto e plotto la distanza del k-esimo vicino, in ordine crescente;
        - trovo il punto di Elbow e plotto il punto di Elbow;
        - uso il punto di Elbow come valore di eps per DBScan;

### Esercizio 4 - Probabilistic Models
Implementazione dei principali modelli di Probabilistic Models basati su Naive Bayes: Bernoulli Multivariato e Bernoulli Multinomiale.

- [Nella parte introduttiva]: viene letto il file di dati e viene caricata, in modo sparso, la matrice utilizzando la libreria scipy.sparse

- [Nella parte da implementare]:
    - viene implementato il modello Bernoulli Multivariato e valutato;
    - viene implementato il modello Bernoulli Multinomiale e valutato;
    - viene implementato il t-test per confrontare i due modelli e capire quanto le differenze siano statisticamente significative.