"""Starter species list and a seed confusable-species-pairs list (stretch goal from the scope
doc, pulled forward into the core build since it's what exercises the reasoning layer's
ambiguity-handling). Expand both once the pipeline is proven — see scope doc section 8.
"""

DEFAULT_SPECIES = [
    "Willow Flycatcher", "Alder Flycatcher",
    "Chipping Sparrow", "American Tree Sparrow",
    "Downy Woodpecker", "Hairy Woodpecker",
    "House Finch", "Purple Finch",
    "Cooper's Hawk", "Sharp-shinned Hawk",
    "Marsh Wren", "Sedge Wren",
    "Least Flycatcher", "Acadian Flycatcher",
    "American Crow", "Common Raven",
    "Black-capped Chickadee", "Carolina Chickadee",
    "Song Sparrow", "Savannah Sparrow",
    "Northern Cardinal", "Scarlet Tanager",
    "Red-winged Blackbird", "European Starling",
]

CONFUSABLE_PAIRS = [
    {"species_a": "Willow Flycatcher", "species_b": "Alder Flycatcher", "type": "acoustic",
     "note": "Near-identical in plumage and call; historically lumped as Traill's Flycatcher, split mainly by voice."},
    {"species_a": "Chipping Sparrow", "species_b": "American Tree Sparrow", "type": "visual",
     "note": "Similar rufous cap; separated mainly by bill color and central breast spot."},
    {"species_a": "Downy Woodpecker", "species_b": "Hairy Woodpecker", "type": "visual",
     "note": "Nearly identical plumage; size and bill-length are the only reliable field cues."},
    {"species_a": "House Finch", "species_b": "Purple Finch", "type": "visual",
     "note": "Both red-and-brown finches; Purple Finch's color is more raspberry/washed, House Finch more streaky."},
    {"species_a": "Cooper's Hawk", "species_b": "Sharp-shinned Hawk", "type": "visual",
     "note": "Classic accipiter ID problem; separated mainly by size and tail shape."},
    {"species_a": "Marsh Wren", "species_b": "Sedge Wren", "type": "acoustic",
     "note": "Both give rapid chattering songs in wetland habitat; Sedge Wren's is drier and less liquid."},
    {"species_a": "Least Flycatcher", "species_b": "Acadian Flycatcher", "type": "acoustic",
     "note": "Empidonax flycatchers — genuinely hard even for experts without voice or habitat context."},
    {"species_a": "Black-capped Chickadee", "species_b": "Carolina Chickadee", "type": "acoustic",
     "note": "Songs/calls overlap heavily in their contact zone; some individuals are unidentifiable by ear alone."},
]
