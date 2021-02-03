# Graphene Automated Detection

Robert Sato
February 2, 2021

## Generalizations and Specifications
- work on single images for now. convery to microscope input later
- saving position is critical but not to be implemented here
- measure mode background HSV/RGB value and set that as background threshold
   - could be useful in imaging in different lighting conditions (possibly irrelevant)
- need some way to find the area of our flakes to avoid saving tiny flakes

### Steps to implement (from Nature paper)
- convert to grayscale
- apply edge detection
- morphology operations to fill the regions surrounded by edges
- calculate the entropy of the grayscale image for each region
- take the intersection between crystals that passed the color-threshold and entropy-threshold filters
- if area is larger than threshold, save the 2D crystal flake

## References
Autonomous robotic searching and assembly of two-dimensional crystals to build van der Waals superlattices - https://www.nature.com/articles/s41467-018-03723-w
