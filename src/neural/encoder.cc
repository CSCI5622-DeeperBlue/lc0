/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include "neural/encoder.h"

#include <algorithm>

namespace lczero {

namespace {

int CompareTransposing(BitBoard board, int initial_transform) {
  uint64_t value = board.as_int();
  if ((initial_transform & FlipTransform) != 0) {
    value = ReverseBitsInBytes(value);
  }
  if ((initial_transform & MirrorTransform) != 0) {
    value = ReverseBytesInBytes(value);
  }
  auto alternative = TransposeBitsInBytes(value);
  if (value < alternative) return -1;
  if (value > alternative) return 1;
  return 0;
}

int ChooseTransform(const ChessBoard& board) {
  // If there are any castling options no transform is valid.
  // Even using FRC rules, king and queen side castle moves are not symmetrical.
  if (!board.castlings().no_legal_castle()) {
    return 0;
  }
  auto our_king = (board.kings() & board.ours()).as_int();
  int transform = NoTransform;
  if ((our_king & 0x0F0F0F0F0F0F0F0FULL) != 0) {
    transform |= FlipTransform;
    our_king = ReverseBitsInBytes(our_king);
  }
  // If there are any pawns only horizontal flip is valid.
  if (board.pawns().as_int() != 0) {
    return transform;
  }
  if ((our_king & 0xFFFFFFFF00000000ULL) != 0) {
    transform |= MirrorTransform;
    our_king = ReverseBytesInBytes(our_king);
  }
  // Our king is now always in bottom right quadrant.
  // Transpose for king in top right triangle, or if on diagonal whichever has
  // the smaller integer value for each test scenario.
  if ((our_king & 0xE0C08000ULL) != 0) {
    transform |= TransposeTransform;
  } else if ((our_king & 0x10204080ULL) != 0) {
    auto outcome = CompareTransposing(board.ours() | board.theirs(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.ours(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.kings(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.queens(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.rooks(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.knights(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.bishops(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    // If all piece types are symmetrical and ours is symmetrical and
    // ours+theirs is symmetrical, everything is symmetrical, so transpose is a
    // no-op.
  }
  return transform;
}
}  // namespace

bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION;
}
bool IsCanonicalArmageddonFormat(
    pblczero::NetworkFormat::InputFormat input_format) {
  return input_format ==
             pblczero::NetworkFormat::
                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == pblczero::NetworkFormat::
                             INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES;
}
bool Is960CastlingFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >= pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE;
}

int TransformForPosition(pblczero::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history) {
  if (!IsCanonicalFormat(input_format)) {
    return 0;
  }
  const ChessBoard& board = history.Last().GetBoard();
  return ChooseTransform(board);
}

InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out) {
  InputPlanes result(kAuxPlaneBase + 8);

  int transform = 0;
  // Canonicalization format needs to stop early to avoid applying transform in
  // history across incompatible transitions.  It is also more canonical since
  // history before these points is not relevant to the final result.
  bool stop_early = IsCanonicalFormat(input_format);
  // When stopping early, we want to know if castlings has changed, so capture
  // it for the first board.
  ChessBoard::Castlings castlings;
  {
    const ChessBoard& board = history.Last().GetBoard();
    const bool we_are_black = board.flipped();
    if (IsCanonicalFormat(input_format)) {
      transform = ChooseTransform(board);
    }
    switch (input_format) {
      case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE: {
        // "Legacy" input planes with:
        // - Plane 104 (0-based) filled with 1 if white can castle queenside.
        // - Plane 105 filled with ones if white can castle kingside.
        // - Plane 106 filled with ones if black can castle queenside.
        // - Plane 107 filled with ones if white can castle kingside.
        if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
        if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
        if (board.castlings().they_can_000()) {
          result[kAuxPlaneBase + 2].SetAll();
        }
        if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
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
        // - Plane 104 for positions of rooks (both white and black) which
        // have
        // a-side (queenside) castling right.
        // - Plane 105 for positions of rooks (both white and black) which have
        // h-side (kingside) castling right.
        const auto& cast = board.castlings();
        result[kAuxPlaneBase + 0].mask =
            ((cast.we_can_000() ? BoardSquare(ChessBoard::A1M).as_board() : 0) |
             (cast.they_can_000() ? BoardSquare(ChessBoard::A8M).as_board() : 0))
            << cast.queenside_rook();
        result[kAuxPlaneBase + 1].mask =
            ((cast.we_can_00() ? BoardSquare(ChessBoard::A1M).as_board() : 0) |
             (cast.they_can_00() ? BoardSquare(ChessBoard::A8M).as_board() : 0))
            << cast.kingside_rook();
        break;
      }
      default:
        throw Exception("Unsupported input plane encoding " +
                        std::to_string(input_format));
    };
    if (IsCanonicalFormat(input_format)) {
      result[kAuxPlaneBase + 4].mask = board.en_passant().as_int();
    } else {
      if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
    }
    if (IsHectopliesFormat(input_format)) {
      result[kAuxPlaneBase + 5].Fill(history.Last().GetRule50Ply() / 100.0f);
    } else {
      result[kAuxPlaneBase + 5].Fill(history.Last().GetRule50Ply());
    }
    // Plane kAuxPlaneBase + 6 used to be movecount plane, now it's all zeros
    // unless we need it for canonical armageddon side to move.
    if (IsCanonicalArmageddonFormat(input_format)) {
      if (we_are_black) result[kAuxPlaneBase + 6].SetAll();
    }
    // Plane kAuxPlaneBase + 7 is all ones to help NN find board edges.
    result[kAuxPlaneBase + 7].SetAll();
    if (stop_early) {
      castlings = board.castlings();
    }
  }
  bool skip_non_repeats =
      input_format ==
          pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
      input_format == pblczero::NetworkFormat::
                          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
  bool flip = false;
  int history_idx = history.GetLength() - 1;
  for (int i = 0; i < std::min(history_planes, kMoveHistory);
       ++i, --history_idx) {
    const Position& position =
        history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    // Castling changes can't be repeated, so we can stop early.
    if (stop_early && board.castlings().as_int() != castlings.as_int()) break;
    // Enpassants can't be repeated, but we do need to always send the current
    // position.
    if (stop_early && history_idx != history.GetLength() - 1 &&
        !board.en_passant().empty()) {
      break;
    }
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
    // Board may be flipped so compare with position.GetBoard().
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        position.GetBoard() == ChessBoard::kStartposBoard) {
      break;
    }
    const int repetitions = position.GetRepetitions();
    // Canonical v2 only writes an item if it is a repeat, unless its the most
    // recent position.
    if (skip_non_repeats && repetitions == 0 && i > 0) {
      if (history_idx > 0) flip = !flip;
      // If no capture no pawn is 0, the previous was start of game, capture or
      // pawn push, so there can't be any more repeats that are worth
      // considering.
      if (position.GetRule50Ply() == 0) break;
      // Decrement i so it remains the same as the history_idx decrements.
      --i;
      continue;
    }

    const int base = i * kPlanesPerBoard;
    // 3d encode pauns
    // Options
    // 1) keep same number of layers, change from 8x8 to 8x8x3? if 4 could 16x16.
    // 2) Have three layers for each type, indicates 112 => 12*3 - 100 = 136?;


    // why board.ours() & board.pawns() instead of board.pawns()?

    // our pawns - bottom, middle, upper
    result[base + 0].mask = board.ours().lower() & board.pawns().lower();
    result[base + 1].mask = board.ours().middle() & board.pawns().middle();
    result[base + 2].mask = board.ours().upper() & board.pawns().upper();
    // our knights - bottom, middle, upper
    result[base + 3].mask = board.ours().lower() & board.knights().lower();
    result[base + 4].mask = board.ours().middle() & board.knights().middle();
    result[base + 5].mask = board.ours().upper() & board.knights().upper();
    // our bishops - bottom, middle, upper
    result[base + 6].mask = board.ours().lower() & board.bishops().lower();
    result[base + 7].mask = board.ours().middle() & board.bishops().middle();
    result[base + 8].mask = board.ours().upper() & board.bishops().upper();
    // our rooks - bottom, middle, upper
    result[base + 9].mask = board.ours().lower() & board.rooks().lower();
    result[base + 10].mask = board.ours().middle() & board.rooks().middle();
    result[base + 11].mask = board.ours().upper() & board.rooks().upper();
    // our queens - bottom, middle, upper
    result[base + 12].mask = board.ours().lower() & board.queens().lower();
    result[base + 13].mask = board.ours().middle() & board.queens().middle();
    result[base + 14].mask = board.ours().upper() & board.queens().upper();
    // our kings - bottom, middle, upper
    result[base + 15].mask = board.ours().lower() & board.kings().lower();
    result[base + 16].mask = board.ours().middle() & board.kings().middle();
    result[base + 17].mask = board.ours().upper() & board.kings().upper();

    // theirs pawns - bottom, middle, upper
    result[base + 18].mask = board.theirs().lower() & board.pawns().lower();
    result[base + 19].mask = board.theirs().middle() & board.pawns().middle();
    result[base + 20].mask = board.theirs().upper() & board.pawns().upper();
    // theirs knights - bottom, middle, upper
    result[base + 21].mask = board.theirs().lower() & board.knights().lower();
    result[base + 22].mask = board.theirs().middle() & board.knights().middle();
    result[base + 23].mask = board.theirs().upper() & board.knights().upper();
    // theirs bishops - bottom, middle, upper
    result[base + 24].mask = board.theirs().lower() & board.bishops().lower();
    result[base + 25].mask = board.theirs().middle() & board.bishops().middle();
    result[base + 26].mask = board.theirs().upper() & board.bishops().upper();
    // theirs rooks - bottom, middle, upper
    result[base + 27].mask = board.theirs().lower() & board.rooks().lower();
    result[base + 28].mask = board.theirs().middle() & board.rooks().middle();
    result[base + 29].mask = board.theirs().upper() & board.rooks().upper();
    // theirs queens - bottom, middle, upper
    result[base + 30].mask = board.theirs().lower() & board.queens().lower();
    result[base + 31].mask = board.theirs().middle() & board.queens().middle();
    result[base + 32].mask = board.theirs().upper() & board.queens().upper();
    // theirs kings - bottom, middle, upper
    result[base + 33].mask = board.theirs().lower() & board.kings().lower();
    result[base + 34].mask = board.theirs().middle() & board.kings().middle();
    result[base + 35].mask = board.theirs().upper() & board.kings().upper();

    if (repetitions >= 1) result[base + 36].SetAll();

    // If en passant flag is set, undo last pawn move by removing the pawn from
    // the new square and putting into pre-move square.
    // 3d update en passant
    if (history_idx < 0 && !board.en_passant().empty()) {
      const auto idx = GetLowestBit(board.en_passant().as_int());
      if (idx < 8) {  // "Us" board
        result[base + 0].mask +=
            ((0x0000000000000100ULL - 0x0000000001000000ULL) << idx);
      } else {
        result[base + 6].mask +=
            ((0x0001000000000000ULL - 0x0000000100000000ULL) << (idx - 56));
      }
    }
    if (history_idx > 0) flip = !flip;
    // If no capture no pawn is 0, the previous was start of game, capture or
    // pawn push, so no need to go back further if stopping early.
    if (stop_early && position.GetRule50Ply() == 0) break;
  }
  if (transform != NoTransform) {
    // Transform all masks.
    for (int i = 0; i <= kAuxPlaneBase + 4; i++) {
      auto v = result[i].mask;
      if (v == 0 || v == ~0ULL) continue;
      if ((transform & FlipTransform) != 0) {
        v = ReverseBitsInBytes(v);
      }
      if ((transform & MirrorTransform) != 0) {
        v = ReverseBytesInBytes(v);
      }
      if ((transform & TransposeTransform) != 0) {
        v = TransposeBitsInBytes(v);
      }
      result[i].mask = v;
    }
  }
  if (transform_out) *transform_out = transform;
  return result;
}

}  // namespace lczero
