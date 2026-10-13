# Stardust or Smog: Stratospheric Dust Origin from One Micrograph

- Domain: Computer Vision
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: image, small-data

---

**Overview**

High-altitude aircraft carry sticky collector plates through the stratosphere to catch the dust that drifts there. Most particles are terrestrial - volcanic ash, rocket-exhaust spheres, industrial debris - and a small share is interplanetary dust from comets and asteroids. Curators record a provisional origin type for each particle. Predict that type from one scanning-electron micrograph:

* cosmic - interplanetary dust: porous grain aggregates, dark Fe-Ni-S spheres, olivine or pyroxene fragments;
* natural\_terrestrial - volcanic ash and mineral fragments;
* artificial\_terrestrial - paint, metal and other man-made debris;
* aluminium\_oxide\_sphere - smooth spheres from solid rocket exhaust.

What sets this task apart:

* The image alone, with the chemistry withheld. Earlier automated classification of these particles worked from their X-ray composition spectra. Here the spectrum and optical notes the curators used are withheld, and the type must be read from shape and surface texture in one greyscale micrograph. Natural and artificial debris can look alike, and the types are provisional, so a perfect score is not available.
* Unseen flights on two instruments. No collector (one flight exposure) is split between training and test: the test set holds whole collectors the model never sees, drawn from both the older micrographs (scans of printed film photographs) and the newer digital ones, and no single collector supplies most of any class.
* Single micron-sized particles caught in flight, not rock or mineral samples: the question is where a particle came from - space, the Earth's surface, industry or rocket exhaust - not what it is made of.

**Dataset**

The public/ folder contains:

* train.csv - columns id, image (file name in train\_images/) and origin.
* test.csv - columns id and image (file name in test\_images/).
* folds.csv - columns id and fold: five folds over the training rows; particles from one collector share a fold.
* sample\_submission.csv - a format placeholder (id, origin) with arbitrary answers; scores about 0.001.
* train\_images/ and test\_images/ - greyscale PNG micrographs, 256 by 256 pixels, one particle near the centre. Magnification varies and is not given.

2,281 training and 715 test particles. Training counts: natural\_terrestrial 827, artificial\_terrestrial 701, cosmic 571, aluminium\_oxide\_sphere 182.

**Submission Format**

A CSV with exactly the columns id and origin, in that order, and one row for each of the 715 ids in test.csv.

```
id,origin
d0123456789ab,cosmic
dfedcba987654,natural_terrestrial
```

origin must be exactly one of aluminium\_oxide\_sphere, artificial\_terrestrial, cosmic, natural\_terrestrial. A submission is REJECTED if its columns differ or any test.csv id is missing, repeated or joined by another id. Any other origin value (blank, misspelled, re-cased) earns no credit for that row.

**Evaluation**

Balanced-accuracy skill, every origin type weighted equally:

```
balanced_accuracy = mean over classes c of ( correct predictions of c / test rows whose class is c )

skill = ( balanced_accuracy - 1/K ) / ( 1 - 1/K ),   K = number of classes present

score = min( 1.0, max( 0.001, skill ) )
```

The score is clipped to the range 0.001 to 1.0, so worse than chance also scores 0.001. A constant answer scores 0.001, and so does the sample submission; random guessing averages about 0.01, and 95% of random submissions score below 0.04.

Direction: Maximize. Min: 0.001. Max: 1.0.

**What Not To Do**

* Do not use external data: no other particle, micrometeorite or dust images, reference collections or type labels.
* Do not try to identify the particles behind the test images or look up their type in any external source, image database or search service.
* Do not use the id, file names or row order as a signal.
* Do not pseudo-label the test set or probe for answers through repeated submissions or side channels.
* Do not pool predictions across test rows (voting, clustering test images, per-group priors): predict each test micrograph on its own.
* Train or fine-tune on the provided training data, within 90 minutes on a single Nvidia A10G GPU, using only the preinstalled packages; public pretrained weights are permitted.
