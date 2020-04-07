# Instructions

Setup a virtual environment (always a good practice),
and activate it. While all Python
versions above 3.5 should work, for easiness in
compatibility the best practice is to use 3.7.

```bash
$ python3.7 -m venv env
$ source env/bin/activate
```

Install necessary libraries for the Lexibank
ecosystem, and after that use
`cldfbench` to clone all the references catalogs
(can take a while; the `-q` flag is just to
silently clone with prompting at each catalog). 

```bash
pip install pylexibank wheel
cldfbench catconfig -q
```

Clone the lexibank datasets we are going to (currently
only WOLD) and install them with `pip` in edit
mode.

```bash
git clone https://github.com/lexibank/wold
pip install -e wold/
```

We can confirm everything is properly installed with
a `pip freeze | grep lexibank`, which should list
`lexibank_wold` installed in edit mode and
`pylexibank`.

Code for loading the data in memory is in
`load_data.py`.
