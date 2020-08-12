# PyBor: A Python library for borrowing detection based on lexical language models

`pybor` is a library to aid in the detection of borrowings in mono- and multi-lingual
concept lists. It implements a variety of statistical methods that can be used
individually or in combination, intended for supporting expert decisions in a
computer-assisted framework.

`pybor` can be used both as a Python library or as a stand-alone, command-line tool.
It is distributed along with example data that can be used to bootstrap other
analyses.

## Installation and usage

Upon release, it will be possible to install the library as any standard Python
package with `pip`. Currently, the library must be installed from source:

```bash
$ pip install .
```

To use the default `wold` package, you will also need to install it in edit mode:

```bash
$ pip install -e wold/
```

Detailed instructions can be found in the [docs](official documentation).

## Changelog

Version 0.1:

  - First public release

## Roadmap

Version 1.0:
  - Published version

## Community guidelines

While the authors can be contacted directly for support, it is recommended that third
parties use GitHub standard features, such as issues and pull requests, to contribute,
report problems, or seek support.

Contributing guidelines, including a code of conduct, can be found in the
`CONTRIBUTING.md` file.

## Authors and citation

The library is developed by John Miller (ivorydragonspiral@gmail.com),
Tiago Tresoldi (tresoldi@shh.mpg.de), and Johann-Mattis List (list@shh.mpg.de).

TT and JML have received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation
programme (grant agreement
No. [ERC Grant #715618](https://cordis.europa.eu/project/rcn/206320/factsheet/en),
[Computer-Assisted Language Comparison](https://digling.org/calc/).

If you use `pybor`, please cite it as:

> Miller, John; Tresoldi, Tiago; List, Johann-Mattis (2020). PyBor, a Python library for
borrowing detection based on lexical language models. Version 0.1. Jena.

In BibTeX:

```bibtex
@misc{Miller2020pybor,
  author = {Miller, John and Tresoldi, Tiago and List, Johann-Mattis},
  title = {PyBor, a Python library for borrowing detection based on lexical language models. Version 0.1.},
  howpublished = {\url{https://github.com/lingpy/pybor}},
  address = {Jena},
  year = {2020},
}
```
