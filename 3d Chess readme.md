# 3d chess updates

The goals of the 3d udpates are to see if this engine can be modified to play 3-layer chess.

## layer terminology

We call:

the uppermost layer 'upper' or 'u'
the middle layer 'middle' or 'm'
the lower layer 'lower' or 'l'

## 3d FEN

Define FEN string starting position as:

    "8/8/8/8/8/8/8/8/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/8/8/8/8/8/8/8/8 w KQkq - 0 1";## FEN Updates

In this case the uppper layer is blank:

8/8/8/8/8/8/8/8/

The middle layer has the standard opening position:
/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/

And the lower layer is blank:

/8/8/8/8/8/8/8/8

## 3d PGN

we add a third character to the standard pgn notation for the layer

eg: 'e2u'