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
*/

#include "src/neural/encoder.h"

#include <gtest/gtest.h>

namespace lczero {

// 3d likely need update
auto kAllSquaresMask = std::numeric_limits<std::uint64_t>::max();


TEST(EncodePositionForNN, EncodeStartPosition) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  // 3d may need to change pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE to new enum
  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 2, FillEmptyHistory::NO, nullptr);

  InputPlane our_pawns_plane_lower = encoded_planes[0];
  InputPlane our_pawns_plane_middle = encoded_planes[1];
  InputPlane our_pawns_plane_upper = encoded_planes[2];

  auto our_pawns_mask_middle = 0ull;

  for (auto i = 0; i < 8; i++) {
    // First pawn is at square a2 (position 8)
    // Last pawn is at square h2 (position 8 + 7 = 15)
    our_pawns_mask_middle |= 1ull << (8 + i);
  }

  //3d update, apply mask test to all 3 layers
  EXPECT_EQ(our_pawns_plane_lower.value, 0.0f);
  EXPECT_EQ(our_pawns_plane_middle.mask, our_pawns_mask_middle);
  EXPECT_EQ(our_pawns_plane_upper.value, 0.0f);

  InputPlane our_knights_plane_lower = encoded_planes[3];
  InputPlane our_knights_plane_middle = encoded_planes[4];
  InputPlane our_knights_plane_upper = encoded_planes[5];
  EXPECT_EQ(our_knights_plane_lower.value, 0.0f);
  EXPECT_EQ(our_knights_plane_middle.mask, (1ull << 1) | (1ull << 6));
  EXPECT_EQ(our_knights_plane_middle.value, 1.0f);
  EXPECT_EQ(our_knights_plane_upper.value, 0.0f);

  InputPlane our_bishops_plane_lower = encoded_planes[6];
  InputPlane our_bishops_plane_middle = encoded_planes[7];
  InputPlane our_bishops_plane_upper = encoded_planes[8];
  EXPECT_EQ(our_bishops_plane_lower.value, 0.0f);
  EXPECT_EQ(our_bishops_plane_middle.mask, (1ull << 2) | (1ull << 5));
  EXPECT_EQ(our_bishops_plane_middle.value, 1.0f);
  EXPECT_EQ(our_bishops_plane_upper.value, 0.0f);

  InputPlane our_rooks_plane_lower = encoded_planes[9];
  InputPlane our_rooks_plane_middle = encoded_planes[10];
  InputPlane our_rooks_plane_upper = encoded_planes[11];
  EXPECT_EQ(our_rooks_plane_lower.value, 0.0f);
  EXPECT_EQ(our_rooks_plane_middle.mask, 1ull | (1ull << 7));
  EXPECT_EQ(our_rooks_plane_middle.value, 1.0f);
  EXPECT_EQ(our_rooks_plane_upper.value, 0.0f);

  InputPlane our_queens_plane_lower = encoded_planes[12];
  InputPlane our_queens_plane_middle = encoded_planes[13];
  InputPlane our_queens_plane_upper = encoded_planes[14];
  EXPECT_EQ(our_queens_plane_lower.value, 0.0f);
  EXPECT_EQ(our_queens_plane_middle.mask, 1ull << 3);
  EXPECT_EQ(our_queens_plane_middle.value, 1.0f);
  EXPECT_EQ(our_queens_plane_upper.value, 0.0f);

  InputPlane our_kings_plane_lower = encoded_planes[15];
  InputPlane our_kings_plane_middle = encoded_planes[16];
  InputPlane our_kings_plane_upper = encoded_planes[17];
  EXPECT_EQ(our_kings_plane_lower.value, 0.0f);
  EXPECT_EQ(our_kings_plane_middle.mask, 1ull << 4);
  EXPECT_EQ(our_kings_plane_middle.value, 1.0f);
  EXPECT_EQ(our_kings_plane_upper.value, 0.0f);

  // Sanity check opponent's pieces
  // 3d update to their king plane indices [33-35]
  InputPlane their_king_plane_lower = encoded_planes[33];
  InputPlane their_king_plane_middle = encoded_planes[34];
  InputPlane their_king_plane_upper = encoded_planes[35];

  auto their_king_row = 7;
  auto their_king_col = 4;
  EXPECT_EQ(their_king_plane_lower.value, 0.0f);
  EXPECT_EQ(their_king_plane_middle.mask,
            1ull << (8 * their_king_row + their_king_col));
  EXPECT_EQ(their_king_plane_middle.value, 1.0f);
  EXPECT_EQ(their_king_plane_upper.value, 0.0f);


  // Start of game, no history.
  for (int j = 37; j < 105; j++) {
    InputPlane zeroed_history = encoded_planes[j];
    EXPECT_EQ(zeroed_history.mask, 0ull);
  }

  // Auxiliary planes

  // It's the start of the game, so all castlings should be allowed.
  // 3d change, change this to encoded_planes[13*2]
  for (auto i = 0; i < 4; i++) {
    InputPlane can_castle_plane = encoded_planes[13 * 8 + i];
    EXPECT_EQ(can_castle_plane.mask, kAllSquaresMask);
    EXPECT_EQ(can_castle_plane.value, 1.0f);
  }

  InputPlane we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 0.0f);

  // We no longer encode the move count, so that plane should be all zeros
  InputPlane zeroed_move_count_plane = encoded_planes[13 * 8 + 6];
  EXPECT_EQ(zeroed_move_count_plane.mask, 0ull);

  InputPlane all_ones_plane = encoded_planes[13 * 8 + 7];
  EXPECT_EQ(all_ones_plane.mask, kAllSquaresMask);
  EXPECT_EQ(all_ones_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeEndGameFormat1) {
  ChessBoard board;
  PositionHistory history;
  //3d update, change starting fen string
  board.SetFromFen("8/8/8/8/8/8/8/8/3r4/4k3/8/1K6/8/8/8/8/8/8/8/8/8/8/8/8 w - - 0 1");
  history.Reset(board, 0, 1);

  int transform;
  // 3d updates, change encoded planes declaration
  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 2, FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, NoTransform);

  InputPlane our_king_plane_lower = encoded_planes[15];
  InputPlane our_king_plane_middle = encoded_planes[16];
  InputPlane our_king_plane_upper = encoded_planes[17];
  EXPECT_EQ(our_king_plane_lower.value, 0.0f);
  EXPECT_EQ(our_king_plane_middle.mask, 1ull << 33);
  EXPECT_EQ(our_king_plane_middle.value, 1.0f);
  EXPECT_EQ(our_king_plane_upper.value, 0.0f);

  InputPlane their_king_plane_lower = encoded_planes[33];
  InputPlane their_king_plane_middle = encoded_planes[34];
  InputPlane their_king_plane_upper = encoded_planes[35];
  EXPECT_EQ(their_king_plane_lower.value, 0.0f);
  EXPECT_EQ(their_king_plane_middle.mask, 1ull << 52);
  EXPECT_EQ(their_king_plane_middle.value, 1.0f);
  EXPECT_EQ(their_king_plane_upper.value, 0.0f);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
