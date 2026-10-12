# Pockmark Mapping From Sparse Survey Lines

- Domain: Computer Vision
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: image, segmentation

---

**Overview:**  
Pockmarks are crater-like depressions in soft seafloor sediment, formed where gas or pore water escapes. Mapping them matters for seabed stability, cable and pipeline routing and studies of fluid flow, and their outlines are normally drawn on dense depth grids from full-coverage sonar surveys. Much of the seafloor, however, has only been surveyed along separate lines: older single-beam surveys, reconnaissance transects and autonomous-vehicle tracks record depth along narrow paths with gaps in between.

This challenge asks for the full pockmark map of a seafloor tile when only such lines were surveyed. Each case is one 256 x 256 tile of gridded depth in which depth is known only along a set of parallel survey lines; everything between the lines is unobserved. Predict the binary mask of all pockmark pixels in the tile, including the parts of pockmarks that fall between the lines.

Task definition:  
The prediction unit is one tile. For each evaluation tile you return one binary mask of 256 x 256 pixels in which 1 marks pockmark pixels, encoded as described under Submission. Every evaluation tile contains at least one pockmark.

What the inputs are:  
Each tile is an 8-bit grayscale PNG. A pixel value of 0 means the pixel was not observed. Values from 1 to 255 are depth values scaled within the tile, where darker (smaller) values are deeper; the scaling is relative, so values cannot be compared between tiles, and a value of 1 can also mark a cell that had no survey data. The observed pixels lie on parallel survey lines about one pixel wide. The heading, spacing and position of the lines differ from tile to tile, and short pieces of some lines are missing. About one pixel in fifteen is observed.

Supervision and split:  
For every training tile you receive the sparse input, the full-coverage depth tile it was sampled from, and the true pockmark mask; the full-coverage tiles are given for training only and do not exist for evaluation tiles. The split is by survey area: the tiles come from four pockmark fields, the training tiles come from three of them and the evaluation tiles from the fourth, so no evaluation tile shares its survey area, terrain or data quality with a training tile. Tile identifiers are random codes. There are 891 training tiles and 450 evaluation tiles.

**Evaluation:**  
The metric is mean Dice, where HIGHER IS BETTER.

For one tile, let P be the set of pixels the submitted mask marks as pockmark and G the set of true pockmark pixels. The tile score is

Dice = 2 x (number of pixels in both P and G) / (number of pixels in P + number of pixels in G)

The final score is the mean of the tile scores over all evaluation tiles. An empty submitted mask scores 0 on a tile, since every evaluation tile contains a pockmark.

Why this metric: the product is a map of where pockmarks are, so the score should measure how well the submitted outline overlaps the true outline, counting both missed pockmark area and false area. Dice does exactly that, treats both kinds of error symmetrically and, unlike pixel accuracy, is not dominated by the large background, which covers most of each tile. Averaging over tiles gives every tile the same weight, so tiles crowded with pockmarks do not outweigh tiles with a single small one.

Floor and ceiling: a mask identical to the true mask scores 1 on a tile, and a mask that shares no pixel with it scores 0. The score therefore runs from 0 to 1. Marking every pixel of every tile scores well below 0.5 because pockmarks cover only a minority of each tile.

**Dataset:**  
The following files are provided in public/.

train.csv - one row per training tile, 891 rows.  
tile\_id - text - opaque identifier of the tile.  
image\_file - text - path of the sparse input image, relative to public/.  
test.csv - one row per evaluation tile, 450 rows, with exactly the same columns as train.csv in the same order.  
train\_labels.csv - the true mask of every training tile. Join on tile\_id.  
tile\_id - text - identifier matching train.csv.  
mask\_rle - text - the pockmark mask, run-length encoded as described under Submission.  
train\_images/ - the sparse input of every training tile, as <tile\_id>.png.  
train\_full\_images/ - the full-coverage depth tile from which each training input was sampled, as <tile\_id>.png, with the same scaling as the input.  
test\_images/ - the sparse input of every evaluation tile, as <tile\_id>.png.  
sample\_submission.csv - a correctly formatted submission that gives every tile the same fixed square mask.  
tile\_id - text - identifier matching test.csv.  
mask\_rle - text - the submitted mask.

Submission:  
Submit a CSV with these two columns.

tile\_id - text - one row for each tile\_id in test.csv, 450 rows plus a header row.  
mask\_rle - text - the predicted mask as run-length pairs: the 256 x 256 mask is read row by row from the top-left pixel, pixels are numbered from 1 to 65,536, and each pair "start length" marks length consecutive pockmark pixels beginning at pixel number start. Pairs are separated by spaces. An empty cell means no pockmark pixel.

Example, with fabricated ids:

tile\_id,mask\_rle  
t0a1b2c3d4e,4100 12 4356 14 4612 15  
t5f6a7b8c9d,30001 6 30257 8

Requirements: a header row and one row per evaluation tile, in any order. Runs may overlap and are clipped to the tile. A file that leaves out any evaluation tile, a duplicated tile\_id, a missing tile\_id or mask\_rle column, or a file that shares no tile\_id with the evaluation set is rejected.

**Rules:**  
The only valid input signal is the pixels of each tile's sparse input image, together with what is learned from the training tiles, their full-coverage tiles and their masks.

The following approaches are not allowed:

* Looking up the source. These tiles come from published seafloor surveys. Searching for, downloading or matching against any bathymetry, any map of pockmarks or any other copy of these surveys or their annotations is not allowed. The only data is the files in public/.
* Using tile\_id strings, file names, file sizes or the row order of any file as a prediction signal. They carry no information about the masks.
* Reassembling tiles. Arranging evaluation tiles into a larger mosaic, matching them against each other or against training tiles, or using one evaluation tile to predict another is not allowed. Each tile is one input.
* Producing masks with hand-written rules. The mask must come from an image model trained on the training tiles. Interpolation between the lines, local depression measures or morphological operations may be inputs to that model or light post-processing tuned on training tiles, but hand-written rules may not decide, filter or override the mask; a solution that keeps most of its score with the trained model removed is rule-based and not allowed.
* Fitting on evaluation inputs in any form: no self-supervised training on evaluation images, pseudo-labelling or test-time adaptation. Thresholds and post-processing are tuned on training tiles only.
* Hardcoding outputs for specific evaluation tiles. Every submitted mask must be the unedited output of the trained model's inference over that tile; drawing or editing masks by hand, or keeping any lookup keyed by tile\_id, is hand-labelling.
* Using private, role-gated or API-key-based models, or calling any external inference API.
* Using non-reproducible external weights or artifacts that are not publicly available.

**Pretrained model policy:**  
Publicly available pretrained image models, such as convolutional or transformer backbones for segmentation, may be used provided their own weights are fine-tuned on the training tiles; freezing some layers is acceptable. Features from a frozen pretrained model fed to a separately trained head do not count as fine-tuning that model. A pretrained model used without training on the supplied tiles does not count as the trained model.
