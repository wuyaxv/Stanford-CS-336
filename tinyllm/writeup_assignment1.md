# Write of Assignment 1

## Problem (`unicode1`): Understanding Unicode

> (a) What Unicode charater does chr(0) return?

A: It returns a zero character `'\x00'`

> (b) How does this character's string representation (`__repr__()`) differ from it's printed representation?

A: It escaped `\` characters and preserved the `'` character.

> (c) What happens when this character occurs in text? ...

A: It prints nothing in representation.

---

## Problem (`unicode2`): Unicode Encoding

> (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

A: Comparing the three encoding machanism, it's not hard to find utf-16 and utf-32 encoded bytes are much longer than utf-8, leading to a longer sequence for training.

> (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

A: A example input could be "你好世界" encoded in utf-8 which is `b'\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c'` Because `[bytes([b])].deocde("utf-8") for b in bytestring` decodes bytes singularly, which is a false way to do so. 

> (c) Give a two byte sequence that does not decode to any Unicode character(s).

A: `'\x80\x00'`. bytes starting with `\x80` can't be decode in utf-8k
