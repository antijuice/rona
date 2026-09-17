# Parameter files

`rna_turner2004.par` is the Turner 2004 nearest-neighbour parameter set in the
`RNAfold parameter file v2.0` layout, the de-facto interchange format for these
tables. The thermodynamic values are the published Turner-laboratory
measurements; the file format is the one used by the ViennaRNA package.

`rona.energy.paramfile` parses it in full, including the padding needed to lift
the file's `CG`-based pair indexing and `int22`'s reduced base alphabet onto the
uniform encoding used throughout `rona`.

To use a different parameter set, pass any file in the same format:

```python
from rona.energy import paramfile
from rona.energy.model import NearestNeighbourModel

model = NearestNeighbourModel(paramfile.load("rna_turner1999.par"))
```
