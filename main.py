"""
main.py
Version Python du script MATLAB "main.m"
"""

import os
import re
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes


def main():
    # ============================================================
    # TP RECONNAISSANCE CODE POSTAL
    # REPONSE STRUCTUREE QUESTION PAR QUESTION
    # Compatible Python
    # ============================================================

    cfg = {
        "trainDir": os.path.join(os.getcwd(), "data", "train"),
        "testDir": os.path.join(os.getcwd(), "data", "test"),
        "minObjectArea": 20,
        "expectedDigits": 5,
        "resizeDigitTo": (64, 64),
        "k": 3,
        "classifier": "knn",
    }

    # ============================================================
    # QUESTION 1
    # Construire Xtrain et Ytrain (apprentissage hors ligne)
    # ============================================================

    print("\n=== QUESTION 1 : APPRENTISSAGE ===")

    Xtrain, Ytrain = build_training_set(cfg)

    print(f"Nombre d'echantillons : {Xtrain.shape[0]}")

    if Xtrain.shape[0] == 0:
        raise RuntimeError("Aucun echantillon d'apprentissage n'a ete extrait.")

    # ============================================================
    # QUESTION 2
    # Calcul des moyennes par classe
    # ============================================================

    print("\n=== QUESTION 2 : MOYENNES PAR CLASSE ===")

    classMeans = compute_class_means(Xtrain, Ytrain)

    print("Moyennes calculees.")

    # ============================================================
    # NORMALISATION
    # ============================================================

    mu = np.mean(Xtrain, axis=0)
    sigma = np.std(Xtrain, axis=0)
    sigma[sigma < np.finfo(float).eps] = 1.0

    XtrainNorm = (Xtrain - mu) / sigma
    classMeansNorm = (classMeans - mu) / sigma

    model = {
        "XtrainNorm": XtrainNorm,
        "Ytrain": Ytrain,
        "classMeansNorm": classMeansNorm,
        "mu": mu,
        "sigma": sigma,
        "k": cfg["k"],
    }

    # ============================================================
    # QUESTION 3
    # Classification en ligne (un code postal)
    # ============================================================

    print("\n=== QUESTION 3 : TEST SUR IMAGE ===")

    fileList = sorted(
        [f for f in os.listdir(cfg["testDir"]) if f.lower().endswith(".png")]
    )

    for fname in fileList:
        imgPath = os.path.join(cfg["testDir"], fname)
        predCode = recognize_postal_code(imgPath, model, cfg)
        print(f"{fname} -> {predCode}")

    # ============================================================
    # QUESTION 4
    # Comparaison PPV vs plus proche moyenne
    # ============================================================

    print("\n=== QUESTION 4 : COMPARAISON ===")

    cfg["classifier"] = "knn"
    res_knn = evaluate(cfg, model)

    cfg["classifier"] = "mean"
    res_mean = evaluate(cfg, model)

    print(f"KNN accuracy : {100 * res_knn:.2f} %")
    print(f"Mean accuracy : {100 * res_mean:.2f} %")

    # ============================================================
    # QUESTION 5
    # Discussion (à mettre dans le rapport)
    # ============================================================
    # -> A FAIRE DANS LE PDF, PAS DANS LE CODE

    # ============================================================
    # QUESTION 6
    # Test avec base indépendante
    # ============================================================

    print("\n=== QUESTION 6 : TEST BASE ===")

    cfg["classifier"] = "knn"
    acc = evaluate(cfg, model)

    print(f"Accuracy test : {100 * acc:.2f} %")

    # ============================================================
    # QUESTION 7
    # Ajout de descripteurs invariants (regionprops)
    # ============================================================
    # -> deja inclus dans extract_features()

    # ============================================================
    # QUESTION 8
    # Correction de rotation
    # ============================================================
    # -> non implementee explicitement ici

    # ============================================================
    # QUESTION 9
    # Separation des chiffres collés
    # ============================================================
    # -> segmentation simple par composantes connexes


# ============================================================
# CONSTRUCTION DATASET
# ============================================================

def build_training_set(cfg):
    files = sorted(
        [f for f in os.listdir(cfg["trainDir"]) if f.lower().endswith(".png")]
    )

    Xtrain = []
    Ytrain = []

    for fname in files:
        filePath = os.path.join(cfg["trainDir"], fname)

        # la classe est le premier caractere du nom de fichier
        try:
            digitClass = int(fname[0])
        except ValueError:
            continue

        I = read_image(filePath)
        digits = segment_digits(I, cfg)

        for digit in digits:
            feat = extract_features(digit)
            Xtrain.append(feat)
            Ytrain.append(digitClass)

    return np.asarray(Xtrain, dtype=float), np.asarray(Ytrain, dtype=int)


# ============================================================
# MOYENNES
# ============================================================

def compute_class_means(X, Y):
    means = np.zeros((10, X.shape[1]), dtype=float)

    for c in range(10):
        idx = (Y == c)
        if np.any(idx):
            means[c, :] = np.mean(X[idx, :], axis=0)

    return means


# ============================================================
# RECONNAISSANCE
# ============================================================

def recognize_postal_code(imgPath, model, cfg):
    I = read_image(imgPath)
    digits = segment_digits(I, cfg)

    code = ""

    for digit in digits:
        feat = extract_features(digit)
        feat = (feat - model["mu"]) / model["sigma"]

        if cfg["classifier"] == "knn":
            lbl = knn(feat, model)
        else:
            lbl = nearest_mean(feat, model)

        code += str(lbl)

    return code


# ============================================================
# KNN
# ============================================================

def knn(x, model):
    D = np.sum((model["XtrainNorm"] - x) ** 2, axis=1)
    idx = np.argsort(D)

    k = min(model["k"], len(idx))
    neigh = model["Ytrain"][idx[:k]]

    values, counts = np.unique(neigh, return_counts=True)
    label_value = values[np.argmax(counts)]

    return int(label_value)


# ============================================================
# PLUS PROCHE MOYENNE
# ============================================================

def nearest_mean(x, model):
    D = np.sum((model["classMeansNorm"] - x) ** 2, axis=1)
    idx = np.argmin(D)
    return int(idx)


# ============================================================
# SEGMENTATION
# ============================================================

def segment_digits(I, cfg):
    I = rgb2gray_if_needed(I)

    # Binarisation type Otsu
    level = otsu_threshold(I)
    BW = I < level

    # On veut les chiffres en blanc
    if np.mean(BW) > 0.5:
        BW = ~BW

    BW = remove_small_objects(BW.astype(bool), cfg["minObjectArea"])
    BW = binary_fill_holes(BW)

    L = label(BW)
    props = regionprops(L)

    digits = []
    boxes = []
    labels_kept = []

    for p in props:
        if p.area >= cfg["minObjectArea"]:
            minr, minc, maxr, maxc = p.bbox
            boxes.append([minc, minr, maxc - minc, maxr - minr])
            labels_kept.append(p.label)

    # Trier les chiffres de gauche à droite
    if len(boxes) > 0:
        boxes = np.asarray(boxes)
        ord_idx = np.argsort(boxes[:, 0])
        labels_kept = [labels_kept[i] for i in ord_idx]

    for lbl in labels_kept:
        mask = (L == lbl)

        # Rogner au plus juste
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        mask = mask[r0:r1, c0:c1]

        # Redimensionnement propre
        mask = resize(
            mask.astype(float),
            cfg["resizeDigitTo"],
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5

        mask = mask.astype(bool)
        digits.append(mask)

    return digits


# ============================================================
# FEATURES
# ============================================================

def extract_features(BW):
    BW = BW.astype(bool)
    BW = remove_small_objects(BW, 5)

    L = label(BW)
    props0 = regionprops(L)

    if len(props0) == 0:
        return np.zeros(5, dtype=float)

    # garder la plus grande composante
    largest = max(props0, key=lambda p: p.area)
    BW = (L == largest.label)

    props = regionprops(label(BW))

    if len(props) == 0:
        return np.zeros(5, dtype=float)

    s = props[0]

    area = float(s.area)
    perim = float(s.perimeter)
    eccentricity = float(s.eccentricity)
    extent = float(s.extent)
    solidity = float(s.solidity)

    feat = np.array([
        area,
        perim,
        eccentricity,
        extent,
        solidity
    ], dtype=float)

    return feat


# ============================================================
# UTILS
# ============================================================

def read_image(path):
    return np.array(Image.open(path))


def rgb2gray_if_needed(I):
    I = np.asarray(I)

    if I.ndim == 3:
        I = rgb2gray(I)

    I = I.astype(float)

    if I.max() > 1:
        I = I / 255.0

    return I


def otsu_threshold(I):
    """
    Version simple et robuste d'Otsu pour éviter les problèmes de libs.
    Entrée attendue dans [0,1].
    """
    I = np.asarray(I, dtype=float)
    I = np.clip(I, 0.0, 1.0)
    I8 = (I * 255).astype(np.uint8)

    hist = np.bincount(I8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()

    if total == 0:
        return 0.5

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    sigma_b2 = np.zeros(256, dtype=np.float64)
    denom = omega * (1.0 - omega)
    valid = denom > 0
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    thresh = np.argmax(sigma_b2)
    return thresh / 255.0


# ============================================================
# EVALUATION
# ============================================================

def evaluate(cfg, model):
    files = sorted(
        [f for f in os.listdir(cfg["testDir"]) if f.lower().endswith(".png")]
    )

    if len(files) == 0:
        return 0.0

    correct = 0
    validCount = 0

    for fname in files:
        trueCode = re.findall(r"\d{5}", fname)

        if len(trueCode) == 0:
            continue

        trueCode = trueCode[0]

        pred = recognize_postal_code(
            os.path.join(cfg["testDir"], fname),
            model,
            cfg
        )

        validCount += 1

        if pred == trueCode:
            correct += 1

    if validCount == 0:
        return 0.0
    else:
        return correct / validCount


if __name__ == "__main__":
    main()