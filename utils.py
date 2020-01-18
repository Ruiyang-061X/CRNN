

class Converter:

    def __init__(self, alphabet, ignore_case=True):
        self.ignore_case = ignore_case
        if self.ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, x):
        if isinstance(x, str):
            if self.ignore_case:
                x = x.lower()
            tmp = []
            for char in x:
                if char not in self.dict:
                    char = '*'
                tmp += [self.dict[char]]
            y = tmp
            length = [len(y)]
        else:
            length = [len(i) for i in x]
            x = ''.join(x)
            y, _ = self.encode(x)

        return y, length

    def decode(self, y, length, raw=False):
        if len(length) == 1:
            length = length[0]
            if raw:
                x = ''.join([self.alphabet[i - 1] for i in y])
            else:
                tmp = []
                for i in range(length):
                    if y[i] != 0 and (not (i > 0 and y[i - 1] == y[i])):
                        tmp.append(self.alphabet[y[i] - 1])
                x = ''.join(tmp)
        else:
            x = []
            i = 0
            for j in length:
                x += [self.decode(y[i : i + j], [j], raw=raw)]
                i += j

        return x