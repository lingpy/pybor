import pickle
from statistics import mean
from tabulate import tabulate
from pybor.data import LexibankDataset

try:
    with open('wold.bin', 'rb') as f:
        lex = pickle.load(f)
except:
    lex = LexibankDataset(
            'wold',
            transform={
                "Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0}
            )
    with open('wold.bin', 'wb') as f:
        pickle.dump(lex, f)

mytable = []
out = open('stats.tsv', 'w')
for language in lex.languages.values():
    print(language['Name'])
    table = lex.get_table(
            language=language['Name'],
            form='Segments',
            classification='Loan'
            )
    # count borrowings
    b = len([x for x in table if x[-1] == 1])
    n = len([x for x in table if x[-1] == 0])
    
    # count sounds
    bs, ns = set(), set()
    for row in table:
        for sound in row[1].split():
            if row[-1] == 1:
                bs.add(sound)
            else:
                ns.add(sound)
    mytable += [[
        language['Name'],
        b/len(table),
        n/len(table),
        len(bs.difference(ns))/len(bs.union(ns)),
        len(ns.difference(bs))/len(bs.union(ns))
        ]]
    out.write(language['Name']+'\t'+'\t'.join(['{0:.2f}'.format(x) for x in
        mytable[-1][1:]])+'\n')
out.close()

print(tabulate(sorted(mytable), headers=[
    'Language', 'BorrW', 'NatW', 'UBorrS',
    'UNatS'], floatfmt='.2f'))
