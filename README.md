# Reconnaissance de code postal par K-NN

Ce projet réalise une reconnaissance simple de chiffres pour lire un code postal à partir d'images.

## Fichiers fournis

- `README.md`
- `main.py`
- `main.ipynb`
- `main.m`

## Structure attendue

```text
projet/
├── main.py
├── main.ipynb
├── main.m
└── data/
    ├── train/
    │   ├── 0_1.png
    │   ├── 0_2.png
    │   ├── ...
    │   └── 9_5.png
    └── test/
        ├── 12345.png
        ├── 59130.png
        ├── 62487.png
        └── ...
```

## Convention de nommage

### Train
Chaque image de `train` doit contenir un seul chiffre.

Le premier chiffre du nom du fichier sert de classe :
- `0_1.png` -> chiffre 0
- `7_3.png` -> chiffre 7
- `9.png` -> chiffre 9

### Test
Chaque image de `test` doit contenir un code postal complet.

Le script lit les 5 chiffres présents dans le nom du fichier comme vérité terrain :
- `62487.png`
- `59130_test.png`
- `12345_1.png`

## Dépendances Python

```bash
pip install numpy pillow scipy scikit-image matplotlib
```

## Lancer la version Python

```bash
python3 main_postal_knn.py
```

## Lancer le notebook

```bash
jupyter notebook
```

Puis ouvrir `main.ipynb`.

## Lancer la version Octave

```octave
pkg load image
main
```

## Pipeline

1. lecture de l'image
2. conversion en niveaux de gris
3. binarisation
4. nettoyage morphologique
5. correction légère de rotation
6. segmentation des chiffres
7. extraction de caractéristiques
8. apprentissage K-NN
9. validation

## Sortie

Le programme affiche :
- le nombre de prototypes appris
- un tableau de synthèse
- le taux de reconnaissance chiffre par chiffre
- le taux de reconnaissance du code complet
