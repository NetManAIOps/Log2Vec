def wordify(instance, i2c):
    return ''.join([i2c[i] for i in instance.chars])

def charseq(word, c2i):
    chars = []
    for c in word:
        if c not in c2i:
            c2i[c] = len(c2i)
        chars.append(c2i[c])
    return chars