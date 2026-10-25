---
title: 'Histopathology Annotation MCP: whole-slide nuclei, cell types and training data from AI assistants'
tags:
  - Python
  - digital pathology
  - histopathology
  - image annotation
  - Model Context Protocol
  - nuclei segmentation
authors:
  - name: Sachin Kale
    orcid: 0000-0000-0000-0000  # TODO: your ORCID
    affiliation: 1
affiliations:
  - name: TODO, your affiliation (or "Independent researcher")
    index: 1
date: 25 October 2026
bibliography: paper.bib
---

# Summary

Histopathology Annotation MCP is an open-source server that lets AI assistants such as Claude annotate and
measure microscopy images, from photographs to whole-slide H&E scans, through the Model Context Protocol
[@mcp]. An assistant can upload a slide, check its quality, outline every nucleus, classify nuclei by cell
type, trace regions of interest from rough boxes, and export the result as training data in COCO, YOLO or
QuPath GeoJSON [@bankhead2017qupath] formats, in one conversation and without a local GPU: the models run on
a hosted service.

The server combines established open models: StarDist [@schmidt2018stardist] finds nuclei, HoVer-Net
[@graham2019hovernet] trained on PanNuke [@gamper2020pannuke] assigns each one a cell type (tumour, immune,
connective tissue, dead or normal epithelial), Segment Anything [@kirillov2023sam] and MedSAM [@ma2024medsam]
trace outlines, Macenko's method [@macenko2009] normalizes stain colours, and HistoQC-style measures
[@janowczyk2019histoqc] flag blur, pen marks and faint staining. Whole slides (Aperio SVS, Hamamatsu NDPI,
pyramidal TIFF) are read tile by tile, so files of several gigabytes never need to fit in memory; an analysis
either samples tissue tiles or covers every tissue tile in parallel and returns a density heatmap.

# Statement of need

Annotated histology images are the raw material of computational pathology, but producing them still means
learning desktop tools, scripting model inference, converting between formats and, for whole slides, managing
GPU resources. Large language model assistants can now call external tools, yet pathology tooling has not been
available to them in a form that is safe to use (destructive operations are marked, results carry their
limitations) and verifiable (the assistant can look at its own outlines).

This server targets three groups: researchers and students who want quantitative measurements (nuclei per
mm², nuclear size, cell-type fractions) without writing code; machine-learning engineers who need consistent
training datasets across many images, with one class list and standard exports; and developers building
pathology agents, who need well-described tools with typed inputs. A browser version
(https://agent.sachin-kale.com/#annotate) exposes the same tools without an assistant.

# Functionality

- **Seeing the image**: `view_image` returns the picture with a pixel grid, optionally with annotations drawn on
  or zoomed into a region, so the assistant can place prompts and check its own outlines.
- **Nuclei**: StarDist with the image resized so nuclei appear at the size the model handles best, from the
  scanner resolution when given, otherwise from the nuclei's own size; sizes in µm² and density per mm².
- **Cell types**: HoVer-Net's type maps read inside each StarDist outline, at full resolution; images
  coarser than about 20x are refused because enlarging them cannot add nuclear detail.
- **Whole slides**: tissue detection, tile sampling or full coverage by parallel workers, per-tile quality,
  a heatmap overview, close-up images of the densest tiles, and every nucleus as QuPath GeoJSON in slide
  coordinates.
- **Datasets**: exports for one image or many, with consistent class identifiers across images.

# Validation

Nuclei detection was evaluated against the expert instance masks of NuInsSeg [@mahbod2024nuinsseg]: 665 H&E
images from 31 human and mouse tissues with 30,698 nuclei. The detection threshold was chosen on the
even-numbered images of each tissue and results are reported on the 332 held-out images. At an intersection
over union of 0.5, F1 rose from 0.47 with the pretrained model's defaults to 0.64 (recall 0.33 to 0.55,
precision 0.81 to 0.78), with panoptic quality 0.48. The benchmark script and per-tissue results are in the
repository. Cell types have not yet been benchmarked by the authors; HoVer-Net's published PanNuke results
are the reference.

# Limitations

Results are for research and education, not diagnosis. The PanNuke-trained HoVer-Net weights are licensed for
non-commercial use (CC BY-NC-SA 4.0). Sampled slide analyses estimate slide-level values from a subset of
tiles; full coverage is available at higher compute cost. Accuracy varies by tissue, and is lowest where
nuclei are sparse, thin or faint.

# Acknowledgements

We thank the authors of StarDist, HoVer-Net and TIAToolbox [@pocock2022tiatoolbox], Segment Anything, MedSAM,
QuPath and NuInsSeg for making their models, code and data openly available.

# References
