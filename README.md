# PyBor: A Python library for borrowing detection based on lexical language models

## Instructions

We recommend strongly to first set up a virtual environment and activate it before testing the code described here. This will guarantee that no version clashes or similar problems occur when installing the dependencies. We developed the library by testing it for Python versions 3.7 and 3.8.

```bash
$ python3 -m venv env
$ source env/bin/activate
```

Install necessary libraries for the Lexibank ecosystem, and after that use
`cldfbench` to clone all the references catalogs (can take a while; the `-q`
flag is just to silently clone without prompting at each catalog). 

```bash
pip install pylexibank wheel
cldfbench catconfig -q
```

Clone the lexibank datasets we are going to use (currently only WOLD) and
install them with `pip` in edit mode.

```bash
git clone https://github.com/lexibank/wold
pip install -e wold/
```

We can confirm everything is properly installed with a `pip freeze | grep
lexibank`, which should list `lexibank_wold` installed in edit mode and
`pylexibank`.


