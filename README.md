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
silently clone without prompting at each catalog). 

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
`load_data.py`. When run, it will print the list of columns in the
`wl` wordlist and the ten first entries:

```bash
$ ▶ python load_data.py
['parameter_id', 'doculect', 'form', 'tokens', 'loan', 'concept']
----------  ----------------  ---------  ---------------------  -----
1_theworld  Swahili           dunia      ɗ u n i a              True
1_theworld  Swahili           ulimwengu  u l i m w e ⁿg u       False
1_theworld  Iraqw             yaamu      j aː m u               False
1_theworld  Gawwada           ʔalame     ʔ a l a m e            True
1_theworld  Hausa             dúuníyàa   d ú/u u n í/i j àa/aː  True
1_theworld  Kanuri            dúnyâ      d ú/u ɲ â/a            True
1_theworld  TarifiytBerber    ddənya     dː ə n y a             True
1_theworld  SeychellesCreole  lemonn     l e m ɔ̃ n              False
1_theworld  Romanian          lume       l u m e                False
----------  ----------------  ---------  ---------------------  -----
```

Tokens might include the slash notation, for which we only care about the
right part. It can be extracted with a simple list comprehension, but
in must cases (like when using methods from `lingpy` or `pyclts`) this is
not necessary.
