"""Decode LabSpec 6 chunk compression (byte shuffle, delta coding, LZHUF).
"""

import struct


_BLOCK_LIMIT = 8 * 1024 * 1024
_OUTPUT_LIMIT = 128 * 1024 * 1024
_POSITION_LENGTHS = (3,) + (4,) * 3 + (5,) * 8 + (6,) * 12 + (7,) * 24 + (8,) * 16
_POSITION_CODES = {}
_code = _previous = 0
for _symbol, _length in enumerate(_POSITION_LENGTHS):
    _code <<= _length - _previous
    _POSITION_CODES[(_length, _code)] = _symbol
    _code += 1
    _previous = _length


class _LzhufDecoder:
    """4 KiB zero-filled dictionary, 60-byte matches, adaptive Huffman tree."""

    _N_CHAR = 314
    _TREE_SIZE = 627
    _ROOT = 626

    def __init__(self, data):
        self.data = data
        self.bit = 0
        self.freq = [1] * 314 + [0] * 313 + [0xffff]
        self.son = list(range(627, 941)) + [0] * 313
        self.parent = [0] * 941
        for i in range(314):
            self.parent[i + 627] = i
        child = 0
        for node in range(314, 627):
            self.freq[node] = self.freq[child] + self.freq[child + 1]
            self.son[node] = child
            self.parent[child] = self.parent[child + 1] = node
            child += 2
        self.parent[self._ROOT] = 0

    def _bits(self, count):
        value = 0
        for _ in range(count):
            if self.bit >= len(self.data) * 8:
                raise ValueError('Truncated LabSpec LZHUF bitstream')
            value = value * 2 + ((self.data[self.bit // 8] >> (7 - self.bit % 8)) & 1)
            self.bit += 1
        return value

    def _reconstruct(self):
        leaves = [((self.freq[i] + 1) // 2, self.son[i])
                  for i in range(self._TREE_SIZE) if self.son[i] >= self._TREE_SIZE]
        for i, (frequency, child) in enumerate(leaves):
            self.freq[i], self.son[i] = frequency, child
        child = 0
        for node in range(self._N_CHAR, self._TREE_SIZE):
            frequency = self.freq[child] + self.freq[child + 1]
            insert = node
            while insert > 0 and frequency < self.freq[insert - 1]:
                insert -= 1
            self.freq[insert + 1:node + 1] = self.freq[insert:node]
            self.son[insert + 1:node + 1] = self.son[insert:node]
            self.freq[insert], self.son[insert] = frequency, child
            child += 2
        for node in range(self._TREE_SIZE):
            child = self.son[node]
            self.parent[child] = node
            if child < self._TREE_SIZE:
                self.parent[child + 1] = node

    def _update(self, symbol):
        if self.freq[self._ROOT] == 0x8000:
            self._reconstruct()
        node = self.parent[symbol + self._TREE_SIZE]
        while True:
            self.freq[node] += 1
            frequency = self.freq[node]
            other = node + 1
            if frequency > self.freq[other]:
                while frequency > self.freq[other]:
                    other += 1
                other -= 1
                self.freq[node], self.freq[other] = self.freq[other], frequency
                child = self.son[node]
                self.parent[child] = other
                if child < self._TREE_SIZE:
                    self.parent[child + 1] = other
                replacement = self.son[other]
                self.son[other] = child
                self.parent[replacement] = node
                if replacement < self._TREE_SIZE:
                    self.parent[replacement + 1] = node
                self.son[node] = replacement
                node = other
            node = self.parent[node]
            if node == 0:
                break

    def _character(self):
        node = self.son[self._ROOT]
        while node < self._TREE_SIZE:
            node = self.son[node + self._bits(1)]
        symbol = node - self._TREE_SIZE
        self._update(symbol)
        return symbol

    def _position(self):
        code = 0
        for length in range(1, 9):
            code = code * 2 + self._bits(1)
            symbol = _POSITION_CODES.get((length, code))
            if symbol is not None:
                return (symbol << 6) | self._bits(6)
        raise ValueError('Invalid LabSpec LZHUF match position')

    def decode(self, size):
        ring = bytearray(4096)
        cursor = 4096 - 60
        deltas = bytearray()
        while len(deltas) < size:
            symbol = self._character()
            if symbol < 256:
                deltas.append(symbol)
                ring[cursor] = symbol
                cursor = (cursor + 1) & 4095
            else:
                source = (cursor - self._position() - 1) & 4095
                length = symbol - 253
                if len(deltas) + length > size:
                    raise ValueError('LabSpec LZHUF match exceeds declared size')
                for i in range(length):
                    value = ring[(source + i) & 4095]
                    deltas.append(value)
                    ring[cursor] = value
                    cursor = (cursor + 1) & 4095
        # Differences are encoded across the entire shuffled block, including
        # transitions between byte planes. The dictionary holds differences.
        previous = 0
        for i, delta in enumerate(deltas):
            previous = (previous + delta) & 255
            deltas[i] = previous
        return deltas


def decompress_labspec6(raw):
    """Return (uncompressed bytes, compression name), preserving the header.

    The first 20 bytes precede chunks of <unpacked_size, packed_size, payload>.
    Equal sizes denote a shuffled block stored without LZHUF/delta coding.
    Both compressed and stored chunks undo a four-plane byte shuffle.
    """
    if len(raw) < 20 or raw[:8] != b'LabSpec6':
        raise ValueError('Not a complete LabSpec 6 header')
    flags = struct.unpack_from('<I', raw, 12)[0]
    if flags & ~5:
        raise ValueError(f'Unsupported LabSpec 6 file flags: 0x{flags:x}')
    if not flags & 1:
        return raw, 'none'
    output = bytearray(raw[:20])
    struct.pack_into('<I', output, 12, flags & ~1)
    pos = 20
    blocks = 0
    while pos < len(raw):
        # LabSpec appends a zero-filled trailer after its final chunk.
        if len(raw) - pos <= 64 and not any(raw[pos:]):
            break
        if pos + 8 > len(raw):
            raise ValueError('Truncated LabSpec compressed chunk header')
        size, packed = struct.unpack_from('<II', raw, pos)
        pos += 8
        if not 0 < size <= _BLOCK_LIMIT or size % 4 or not 0 < packed <= size:
            raise ValueError('Invalid LabSpec compressed chunk sizes')
        if len(output) + size > _OUTPUT_LIMIT:
            raise ValueError('LabSpec spectrum exceeds decompression size limit')
        if pos + packed > len(raw):
            raise ValueError('Truncated LabSpec compressed chunk payload')
        payload = raw[pos:pos + packed]
        pos += packed
        shuffled = _LzhufDecoder(payload).decode(size) if packed < size else payload
        block = bytearray(size)
        plane_size = size // 4
        for plane in range(4):
            block[plane::4] = shuffled[plane * plane_size:(plane + 1) * plane_size]
        output.extend(block)
        blocks += 1
    if not blocks:
        raise ValueError('LabSpec compressed file contains no chunks')
    return bytes(output), 'LZHUF+delta+byte-shuffle'
