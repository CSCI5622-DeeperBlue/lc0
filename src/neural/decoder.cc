/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "neural/decoder.h"

#include "neural/encoder.h"

namespace lczero {

namespace {

// this is hacked up at the moment to compile.

BoardSquare SingleSquare(BitBoard input) {
  for (auto sq : input) {
    return sq;
  }
  assert(false);
  return BoardSquare();
}
// 3d updates: changed arguments to be BitBoards instead of planes
BitBoard MaskDiffWithMirror(const BitBoard& cur, const BitBoard& prev) {
  // prev.Mirror();
  // return cur.xor(prev);
  return cur;
}

BoardSquare OldPosition(const InputPlane& prev, BitBoard mask_diff) {
  // auto to_mirror = BitBoard(prev.mask);
  // to_mirror.Mirror();
  // return SingleSquare(to_mirror & mask_diff);
  return SingleSquare(BitBoard(0,0,0));

}

}  // namespace

void PopulateBoard(pblczero::NetworkFormat::InputFormat input_format,
                   InputPlanes planes, ChessBoard* board, int* rule50,
                   int* gameply) {
  auto pawnsOurs = BitBoard(planes[0].mask, planes[1].mask, planes[2].mask);
  auto knightsOurs = BitBoard(planes[3].mask, planes[4].mask, planes[5].mask);
  auto bishopOurs = BitBoard(planes[6].mask, planes[7].mask, planes[8].mask);
  auto rookOurs = BitBoard(planes[9].mask, planes[10].mask, planes[11].mask);
  auto queenOurs = BitBoard(planes[12].mask, planes[13].mask, planes[14].mask);
  auto kingOurs = BitBoard(planes[15].mask, planes[16].mask, planes[17].mask);

  auto pawnsTheirs =
      BitBoard(planes[18].mask, planes[19].mask, planes[20].mask);
  auto knightsTheirs =
      BitBoard(planes[21].mask, planes[22].mask, planes[23].mask);
  auto bishopTheirs =
      BitBoard(planes[24].mask, planes[25].mask, planes[26].mask);
  auto rookTheirs = BitBoard(planes[27].mask, planes[28].mask, planes[29].mask);
  auto queenTheirs =
      BitBoard(planes[30].mask, planes[31].mask, planes[32].mask);
  auto kingTheirs = BitBoard(planes[33].mask, planes[34].mask, planes[35].mask);

  ChessBoard::Castlings castlings;
  switch (input_format) {
    case pblczero::NetworkFormat::InputFormat::INPUT_CLASSICAL_112_PLANE: {
      if (planes[kAuxPlaneBase + 0].mask != 0) {
        castlings.set_we_can_000();
      }
      if (planes[kAuxPlaneBase + 1].mask != 0) {
        castlings.set_we_can_00();
      }
      if (planes[kAuxPlaneBase + 2].mask != 0) {
        castlings.set_they_can_000();
      }
      if (planes[kAuxPlaneBase + 3].mask != 0) {
        castlings.set_they_can_00();
      }
      break;
    }
    case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
      auto queenside = 0;
      auto kingside = 7;
      if (planes[kAuxPlaneBase + 0].mask != 0) {
        auto mask = planes[kAuxPlaneBase + 0].mask;
        queenside = GetLowestBit((mask >> 56) | mask);
        if ((mask & 0xFFLL) != 0) {
          castlings.set_we_can_000();
        }
        if (mask >> 56 != 0) {
          castlings.set_they_can_000();
        }
      }
      if (planes[kAuxPlaneBase + 1].mask != 0) {
        auto mask = planes[kAuxPlaneBase + 1].mask;
        kingside = GetLowestBit((mask >> 56) | mask);
        if ((mask & 0xFFLL) != 0) {
          castlings.set_we_can_00();
        }
        if (mask >> 56 != 0) {
          castlings.set_they_can_00();
        }
      }
      castlings.SetRookPositions(queenside, kingside);
      break;
    }

    default:
      throw Exception("Unsupported input plane encoding " +
                      std::to_string(input_format));
  }
  std::string fen;
  // Canonical input has no sense of side to move, so we should simply assume
  // the starting position is always white.
  bool black_to_move =
      !IsCanonicalFormat(input_format) && planes[kAuxPlaneBase + 4].mask != 0;
  if (black_to_move) {
    // Flip to white perspective rather than side to move perspective.
    std::swap(pawnsOurs, pawnsTheirs);
    std::swap(knightsOurs, knightsTheirs);
    std::swap(bishopOurs, bishopTheirs);
    std::swap(rookOurs, rookTheirs);
    std::swap(queenOurs, queenTheirs);
    std::swap(kingOurs, kingTheirs);
    pawnsOurs.Mirror();
    pawnsTheirs.Mirror();
    knightsOurs.Mirror();
    knightsTheirs.Mirror();
    bishopOurs.Mirror();
    bishopTheirs.Mirror();
    rookOurs.Mirror();
    rookTheirs.Mirror();
    queenOurs.Mirror();
    queenTheirs.Mirror();
    kingOurs.Mirror();
    kingTheirs.Mirror();
    castlings.Mirror();
  }

  // 3d udpates temporary fix, likely need to iterate from 2 -> 0 with layer and
  // add up fen string
  const int layer = 1;
  for (int layer = 2; layer >= 0; --layer) { // 3d update
    for (int row = 7; row >= 0; --row) {
      int emptycounter = 0;
      for (int col = 0; col < 8; ++col) {
        char piece = '\0';
        if (pawnsOurs.get(row, col, layer)) {
          piece = 'P';
        } else if (pawnsTheirs.get(row, col, layer)) {
          piece = 'p';
        } else if (knightsOurs.get(row, col, layer)) {
          piece = 'N';
        } else if (knightsTheirs.get(row, col, layer)) {
          piece = 'n';
        } else if (bishopOurs.get(row, col, layer)) {
          piece = 'B';
        } else if (bishopTheirs.get(row, col, layer)) {
          piece = 'b';
        } else if (rookOurs.get(row, col, layer)) {
          piece = 'R';
        } else if (rookTheirs.get(row, col, layer)) {
          piece = 'r';
        } else if (queenOurs.get(row, col, layer)) {
          piece = 'Q';
        } else if (queenTheirs.get(row, col, layer)) {
          piece = 'q';
        } else if (kingOurs.get(row, col, layer)) {
          piece = 'K';
        } else if (kingTheirs.get(row, col, layer)) {
          piece = 'k';
        }
        if (emptycounter > 0 && piece) {
          fen += std::to_string(emptycounter);
          emptycounter = 0;
        }
        if (piece) {
          fen += piece;
        } else {
          emptycounter++;
        }
      }
      if (emptycounter > 0) fen += std::to_string(emptycounter);
      if (row > 0) fen += "/";
    }
  }
  fen += " ";
  fen += black_to_move ? "b" : "w";
  fen += " ";
  fen += castlings.as_string();
  fen += " ";
  // 3d updates, changed fen string to
  // fen argument in setfromfen is only has:
  // piece placement, active color, and castling availability (i.e. 8/8/.../8/8 w KQkq)

  //if (IsCanonicalFormat(input_format)) {
  //  // Canonical format helpfully has the en passant details ready for us.
  //  if (planes[kAuxPlaneBase + 4].mask == 0) {
  //    fen += "-";
  //  } else {
  //    int col = GetLowestBit(planes[kAuxPlaneBase + 4].mask >> 56);
  //    fen += BoardSquare(5, col).as_string();
  //  }
  //} else {

  //  auto pawndiff = bitboard(planes[6].mask ^ planes[kplanesperboard + 6].mask);
  //  // if no pawns then 2 pawns, history isn't filled properly and we shouldn't
  //  // try and infer enpassant.
  //  if (pawndiff.count() == 2 && planes[kplanesperboard + 6].mask != 0) {
  //    auto from =
  //        singlesquare(planes[kplanesperboard + 6].mask & pawndiff.as_int());
  //    auto to = singlesquare(planes[6].mask & pawndiff.as_int());
  //    if (from.col() != to.col() || std::abs(from.row() - to.row()) != 2) {
  //      fen += "-";
  //    } else {
  //      // todo: ensure enpassant is legal rather than setting it blindly?
  //      // doesn't matter for rescoring use case as only legal moves will be
  //      // performed afterwards.
  //      fen +=
  //          boardsquare((planes[kauxplanebase + 4].mask != 0) ? 2 : 5, to.col())
  //              .as_string();
  //    }
  //  } else {
  //    fen += "-";
  //  }
  //}
  //fen += " ";
  //int rule50plane = (int)planes[kauxplanebase + 5].value;
  //if (ishectopliesformat(input_format)) {
  //  rule50plane = (int)(100.0f * planes[kauxplanebase + 5].value);
  //}
  //fen += std::to_string(rule50plane);
  //// reuse the 50 move rule as gameply since we don't know better.
  //fen += " ";
  //fen += std::to_string(rule50plane);
  board->SetFromFen(fen, rule50, gameply);
}

// 3d may be able to ignore? check back later
// hacked up for compile.
Move DecodeMoveFromInput(const InputPlanes& planes, const InputPlanes& prior) {
  auto pawndiff = MaskDiffWithMirror(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask), BitBoard(prior[0].mask,prior[1].mask, prior[2].mask)); //MaskDiffWithMirror(BBbp, BBwp);
  auto knightdiff = MaskDiffWithMirror(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask), BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
  auto bishopdiff = MaskDiffWithMirror(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask), BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
  auto rookdiff = MaskDiffWithMirror(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask), BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
  auto queendiff = MaskDiffWithMirror(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask), BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
  // Handle Promotion.
  if (pawndiff.count() == 1) {
    auto from = SingleSquare(pawndiff);
    if (knightdiff.count() == 1) {
      auto to = SingleSquare(knightdiff);
      return Move(from, to, Move::Promotion::Knight);
    }
    if (bishopdiff.count() == 1) {
      auto to = SingleSquare(bishopdiff);
      return Move(from, to, Move::Promotion::Bishop);
    }
    if (rookdiff.count() == 1) {
      auto to = SingleSquare(rookdiff);
      return Move(from, to, Move::Promotion::Rook);
    }
    if (queendiff.count() == 1) {
      auto to = SingleSquare(queendiff);
      return Move(from, to, Move::Promotion::Queen);
    }
    assert(false);
    return Move();
  }
  // check king first as castling moves both king and rook.
  auto kingdiff = MaskDiffWithMirror(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask), BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
  if (kingdiff.count() == 2) {
    if (rookdiff.count() == 2) {
      auto from = OldPosition(prior[5], kingdiff);
      auto to = OldPosition(prior[3], rookdiff);
      return Move(from, to);
    }
    auto from = OldPosition(prior[5], kingdiff);

    // auto to = SingleSquare(planes[11].mask & kingdiff.as_int());
    auto to = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));

    if (std::abs(from.col() - to.col()) > 1) {
      // Chess 960 castling can leave the rook in place, but the king has moved
      // from one side of the rook to the other - thus has gone at least 2
      // squares, which is impossible for a normal king move. Can't work out the
      // rook location from rookdiff since its empty, but it is known given the
      // direction of the king movement and the knowledge that the rook hasn't
      // moved.
      if (from.col() > to.col()) {
        to = BoardSquare(from.row(), to.col() + 1);
      } else {
        to = BoardSquare(from.row(), to.col() - 1);
      }
    }
    return Move(from, to);
  }
  if (queendiff.count() == 2) {
    auto from = OldPosition(prior[4], queendiff);
    auto to = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
    // auto to = SingleSquare(planes[10].mask & queendiff.as_int());

    return Move(from, to);
  }
  if (rookdiff.count() == 2) {
    auto from = OldPosition(prior[3], rookdiff);

    auto to = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
    // Only one king, so we can simply grab its current location directly.

    // auto kingpos = SingleSquare(planes[11].mask);
    auto kingpos = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
    if (from.row() == kingpos.row() && to.row() == kingpos.row() &&
        (from.col() < kingpos.col() && to.col() > kingpos.col() ||
         from.col() > kingpos.col() && to.col() < kingpos.col())) {
      // If the king hasn't moved, this could still be a chess 960 castling move
      // if the rook has passed through the king.
      // Destination of the castling move is where the rook started.
      to = from;
      // And since the king didn't move it forms the start position.
      from = kingpos;
    }
    return Move(from, to);
  }
  if (bishopdiff.count() == 2) {
    auto from = OldPosition(prior[2], bishopdiff);


    // auto to = SingleSquare(planes[8].mask & bishopdiff.as_int());
    auto to = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
    return Move(from, to);
  }
  if (knightdiff.count() == 2) {
    auto from = OldPosition(prior[1], knightdiff);

    auto to = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
    // auto to = SingleSquare(planes[7].mask & knightdiff.as_int());
    return Move(from, to);
  }
  if (pawndiff.count() == 2) {
    auto from = OldPosition(prior[0], pawndiff);
    auto to = SingleSquare(BitBoard(prior[0].mask,prior[1].mask, prior[2].mask));
    // auto to = SingleSquare(planes[6].mask & pawndiff.as_int());

    return Move(from, to);
  }
  assert(false);
  return Move();
}

}  // namespace lczero
