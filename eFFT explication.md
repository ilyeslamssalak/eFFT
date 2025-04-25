## Résumé

La classe **Stimulus** modélise un point d’entrée (row, col) avec un état booléen et définit l’égalité entre stimuli via `__eq__`/`__ne__`.  
La classe **Stimuli** hérite de `list` pour représenter un ensemble de stimuli et fournit deux méthodes : `filter()` pour trier et dédupliquer par coordonnées, et `set_state()` pour uniformiser l’état.  
La classe **EFFT** implémente un FFT bidimensionnel « événementiel » de taille N×N (N puissance de 2). Elle pré­calcule les facteurs de rotation (« twiddle »), construit récursivement un arbre de résultats partiels dans `initialize()`, permet des mises à jour incrémentales via `update()`, et expose le résultat final avec `get_fft()`.

---

## Classe `Stimulus`

### Attributs

- **row** (int) : indice de ligne du point.
    
- **col** (int) : indice de colonne du point.
    
- **state** (bool) : état associé (`True` ou `False`).
    

### Méthodes

- **`__eq__(self, other) -> bool`**  
    Définit la comparaison d’égalité (`==`) entre deux `Stimulus` : deux instances sont égales si leurs `row` et `col` correspondent [GeeksforGeeks](https://www.geeksforgeeks.org/dunder-magic-methods-python/?utm_source=chatgpt.com).
    
- **`__ne__(self, other) -> bool`**  
    Implémente la comparaison d’inégalité (`!=`) en niant le résultat de `__eq__`.
    

---

## Classe `Stimuli`

Hérite de **`list[Stimulus]`** pour gérer un vecteur de stimuli.

### Méthodes

1. **`filter(self)`**
    
    - Trie la liste _in-place_ selon `(row, col, not state)`, de façon à regrouper d’abord par lignes, puis colonnes, et à prioriser `state=True` avant `False` [Python documentation](https://docs.python.org/3/howto/sorting.html?utm_source=chatgpt.com).
        
    - Élime les doublons (mêmes `row` et `col`) en ne gardant que la première occurrence, ce qui permet de ne conserver qu’un seul stimulus par position.
        
2. **`set_state(self, state: bool)`**  
    Parcourt tous les `Stimulus` et leur affecte l’état donné, utile pour réinitialiser ou uniformiser rapidement le vecteur.
    

---

## Classe `EFFT`

Implémente un algorithme FFT 2D récurrent fondé sur la décomposition en quads (butterfly) et la mise à jour incrémentale via des événements.

### Attributs principaux

- **`N`** : taille de la transformée (doit être une puissance de 2).
    
- **`LOG2_N = int(log2(N))`** : nombre d’étages dans l’arbre (log₂(N)) [numpy.org](https://numpy.org/doc/stable/reference/routines.math.html?utm_source=chatgpt.com).
    
- **`tree`** : liste de listes de matrices `np.ndarray[(n,n), complex64]`, un niveau par n = 2ᶦ.
    
- **`twiddle`** : tableau `[i,n] → exp(−2πi·i/n)` pour tous i∈[0,N−1], n∈[1,N], pré­calculé avec `np.exp` et `np.pi` [numpy.org](https://numpy.org/doc/2.2/reference/generated/numpy.exp.html?utm_source=chatgpt.com)[Vultr Docs](https://docs.vultr.com/python/third-party/numpy/exp?utm_source=chatgpt.com).
    

### Méthode `initialize(x=None, offset=0)`

1. **Base** : si `n=1`, stocke une matrice 1×1 dans `tree[0]`.
    
2. **Récursion** : pour n>1, découpe `x` en quatre sous-matrices (quadrants) de taille n/2 × n/2 et appelle `initialize` sur chacune, avec des offsets différents, ce qui construit récursivement les FFT partielles.
    
3. **Butterfly** : une fois les quadrants transformés, combine-les selon l’algorithme standard du FFT 2D :
    
    a=x00+ωnjx01,b=x00−ωnjx01,c=ωnix10+ωni+jx11,d=ωnix10−ωni+jx11,a = x_{00} + \omega_n^j x_{01},\quad b = x_{00} - \omega_n^j x_{01},\quad c = \omega_n^i x_{10} + \omega_n^{i+j} x_{11},\quad d = \omega_n^i x_{10} - \omega_n^{i+j} x_{11},a=x00​+ωnj​x01​,b=x00​−ωnj​x01​,c=ωni​x10​+ωni+j​x11​,d=ωni​x10​−ωni+j​x11​,
    
    puis répartit `(a+c)`, `(b+d)`, `(a−c)`, `(b−d)` dans la matrice résultante.
    
4. Stocke le résultat combiné dans `tree[idx+1]`.
    

Cette structure d’**arbre** permet de conserver tous les niveaux intermédiaires pour des mises à jour locales ultérieures.

> **FFT (Fast Fourier Transform)** : algorithme en O(N log N) pour calculer la DFT, largement utilisé en sciences, ingénierie et traitement du signal [Wikipedia](https://en.wikipedia.org/wiki/Fast_Fourier_transform?utm_source=chatgpt.com).

### Méthode `update(stimuli)`

Permet de modifier l’image d’entrée « au fil de l’eau » sans recomputing global :

- **Single Stimulus** : appel à `_update_matrix` sur la racine `tree[LOG2_N][0]`.
    
- **Multiples Stimuli** : boucle sur chaque `Stimulus` (dans un `Stimuli`) et applique `_update_matrix`.
    

#### Fonction interne `_update_matrix(x, p, offset)`

1. **Cas n=1** : échange la valeur de `x[0,0]` par `p.state` et retourne `True` si l’état a changé.
    
2. **Cas n>1** :
    
    - Identifie le quadrant cible en fonction des bits de `p.row` et `p.col`.
        
    - Appelle récursivement `_update_matrix` sur la sous-matrice correspondante dans `tree[idx][child_offset]`.
        
    - Si un changement a eu lieu, re-combine les quatre quadrants via le même schéma butterfly qu’en initialisation pour remettre à jour la portion affectée de `x`.
        
3. Retourne un booléen indiquant si la FFT a été modifiée.
    

Cette approche **incrémentale** évite de recalculer l’intégralité de la FFT pour chaque petit changement, ce qui est très efficace pour des applications interactives.

### Méthode `get_fft()`

Retourne simplement la matrice finale stockée dans `tree[LOG2_N][0]`, soit le résultat complet de la FFT 2D.