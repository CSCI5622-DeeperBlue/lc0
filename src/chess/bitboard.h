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

#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "utils/bititer.h"

namespace lczero {

// Stores a coordinates of a single square.
class BoardSquare {
 public:
  constexpr BoardSquare() {}
  // As a single number, 0 to 63, bottom to top, left to right, lower to upper
  // 0 is a1l, 8 is a2l, 63 is h8l.
  // 64 is a1m, 71 is a1l, 127 is h8m
  // 128 is a1u, 191 is h8u

  constexpr BoardSquare(std::uint8_t num) : square_(num) {}
  // From row(bottom to top), and col(left to right), 0-based.
  constexpr BoardSquare(int row, int col, int layer) : BoardSquare(64*layer + row * 8 + col) {}

  // might do away with this, assumes if layer isn't specified is actually on middle layer
  constexpr BoardSquare(int row, int col) : BoardSquare(64 + row * 8 + col) {}
  // From Square name, e.g e4. Only lowercase.

  BoardSquare(const std::string& str, bool black = false)
      : BoardSquare(black ? '8' - str[1] : str[1] - '1', str[0] - 'a') {}
  constexpr std::uint8_t as_int() const { return square_; }
  constexpr std::uint64_t as_board() const { return 1ULL << square_; }

  // @param row 0 index row (0  = a, 7 = h)
  // @param col 0 indexed col (0 = 1, 7 = 8)
  // @param layer 0 indexed layer, (0 = b, 2 = u)
  void set(int row, int col, int layer) { square_ = layer*64 + row * 8 + col; }

  // 0-based, bottom to top.
  int row() const { return square_ / 8 ; }
  // 0-based, left to right.
  int col() const { return square_ % 8; }
  // 0-based, lower layer to upper layer.
  int layer() const { return square_ % 64; }

  // Row := 7 - row.  Col remains the same.
  void Mirror() { square_ = square_ ^ 0b111000; }

  // Checks whether coordinate is within 0..7.
  static bool IsValidCoord(int x) { return x >= 0 && x < 8; }

  // Checks whether coordinates are within 0..7.
  static bool IsValid(int row, int col) {
    return row >= 0 && col >= 0 && row < 8 && col < 8;
  }

  constexpr bool operator==(const BoardSquare& other) const {
    return square_ == other.square_;
  }

  constexpr bool operator!=(const BoardSquare& other) const {
    return square_ != other.square_;
  }

  // Returns the square in algebraic notation (e.g. "e4").
  std::string as_string() const {
    return std::string(1, 'a' + col()) + std::string(1, '1' + row());
  }

 private:
  std::uint8_t square_ = 0;  // Only lower six bits should be set.
};

// Represents a board as an array of 64 bits.
// Bit enumeration goes from bottom to top, from left to right:
// Square a1 is bit 0, square a8 is bit 7, square b1 is bit 8.
// 3d we need 64*3 192 bits to store all squares. We can combine three of these in an array or make an update addition/division by overloading the c++ operations
// https://en.cppreference.com/w/cpp/language/operators
// can likely reduce this to just one of a 256 type
// https://stackoverflow.com/questions/5242819/c-128-256-bit-fixed-size-integer-types
class BitBoard {
 public:

  constexpr BitBoard(std::uint64_t board) : board_lower_(board) {}
  constexpr BitBoard(std::uint64_t board) : board_middle_(board) {}
  constexpr BitBoard(std::uint64_t board) : board_upper_(board) {}

  BitBoard() = default;
  BitBoard(const BitBoard&) = default;
  BitBoard(uint64_t lower, uint64_t middle, uint64_t upper);

  //3d todo: might get away with this?
  std::uint64_t as_int() const { return board_lower_ + 64*board_middle_ + 128*board_upper_; }

  void clear() {
    board_lower_ = 0;
    board_middle_ = 0;
    board_upper_ = 0;
   }

  // Counts the number of set bits in the BitBoard.

  // 3d TODO, Hail Mary
  int count() const {
#if defined(NO_POPCNT)
    std::uint64_t x = board_;
    std::uint64_t y = board_;
    std::uint64_t z = board_;
    return NO_POPCNT_helper(x) + NO_POPCNT_helper(y) + NO_POPCNT_helper(z);

#elif defined(_MSC_VER) && defined(_WIN64)
    return  _mm_popcnt_u64(board_lower_) +
            _mm_popcnt_u64(board_middle_) +
            _mm_popcnt_u64(board_upper_);
#elif defined(_MSC_VER)
    return  __popcnt(board_lower_) + __popcnt(board_lower_ >> 32) +
            __popcnt(board_middle_) + __popcnt(board_middle_ >> 32) +
            __popcnt(board_upper_) + __popcnt(board_upper_ >> 32);
#else
    return  __builtin_popcountll(board_lower_) +
            __builtin_popcountll(board_middle_) +
            __builtin_popcountll(board_upper_);
#endif
  }

  // Like count() but using algorithm faster on a very sparse BitBoard.
  // May be slower for more than 4 set bits, but still correct.
  // Useful when counting bits in a Q, R, N or B BitBoard.
  int count_few() const {
#if defined(NO_POPCNT)
    std::uint64_t x = board_;
    int count;
    for (count = 0; x != 0; ++count) {
      // Clear the rightmost set bit.
      x &= x - 1;
    }
    return count;
#else
    return count();
#endif
  }

  // helper for function above
  uint64_t NO_POPCNT_helper(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555;
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
    return (x * 0x0101010101010101) >> 56;
  }

  // Sets the value for given square to 1 if cond is true.
  // Otherwise does nothing (doesn't reset!).
  void set_if(BoardSquare square, bool cond) { set_if(square.as_int(), cond); }
  void set_if(std::uint8_t pos, bool cond) { if(cond) { set(pos); } }
  void set_if(int row, int col, bool cond) {
    set_if(BoardSquare(row, col), cond);
  }

  // Sets value of given square to 1.
  void set(BoardSquare square) { set(square.as_int()); }
  void set(std::uint8_t pos) {
    if(pos < 64) {
      board_lower_ |= (std::uint64_t(1) << pos);
    } else if (pos < 128){
      board_middle_ |= (std::uint64_t(1) << pos);
    } else {
      board_upper_ |= (std::uint64_t(1) << pos);
    }
  }
  void set(int row, int col, int layer) { set(BoardSquare(row, col, layer)); }

  // Sets value of given square to 0.
  void reset(BoardSquare square) { reset(square.as_int()); }
  void reset(std::uint8_t pos) {
    if(pos < 64) {
      board_lower_ &= ~(std::uint64_t(1) << pos);
    } else if (pos < 128){
      board_middle_ &= ~(std::uint64_t(1) << pos);
    } else {
      board_upper_ &= ~(std::uint64_t(1) << pos);
    }
  }

  void reset(int row, int col, int layer) { reset(BoardSquare(row, col, layer)); }

  // Gets value of a square.
  bool get(BoardSquare square) const { return get(square.as_int()); }

  bool get(std::uint8_t pos) const {
    if(pos < 64) {
      return board_lower_ & (std::uint64_t(1) << pos);
    } else if (pos < 128){
      return board_middle_ & (std::uint64_t(1) << pos);
    } else {
      return board_upper_ & (std::uint64_t(1) << pos);
    }
  }

  bool get(int row, int col, int layer) const { return get(BoardSquare(row, col, layer)); }

  // returns pieces on specific layer
  // 3d TODO: these also need to be shifted to 64bits
  int lower() const { return board_lower_ ;}
  // 0-based, left to right.
  int middle() const { return board_middle_;}
  // 0-based, lower layer to upper layer.
  int upper() const { return board_upper_;}

  // Returns whether all bits of a board are set to 0.
  bool empty() const {
    return  (board_lower_ == 0) &&
            (board_middle_ == 0) &&
            (board_upper_ == 0);
  }

  // Checks whether two bitboards have common bits set.
  bool intersects(const BitBoard& other) const {
    return (board_lower_ & other.board_lower_) ||
           (board_middle_ & other.board_middle_) ||
           (board_upper_ & other.board_upper_);
  }

  // Flips black and white side of a board.
  void Mirror() {
    board_lower_ = ReverseBytesInBytes(board_lower_);
    board_middle_ = ReverseBytesInBytes(board_middle_);
    board_upper_ = ReverseBytesInBytes(board_upper_);
  }

  bool operator==(const BitBoard& other) const {
    return  (board_lower_ == other.board_lower_) &&
            (board_middle_ == other.board_middle_) &&
            (board_upper_ == other.board_upper_);
  }

  bool operator!=(const BitBoard& other) const {
    return (board_lower_ != other.board_lower_) ||
           (board_middle_ != other.board_middle_) ||
           (board_upper_ != other.board_upper_);
  }

  // 3d TODO
  BitIterator<BoardSquare> begin() const { return board_lower_; }
  BitIterator<BoardSquare> end() const { return 0; }

  std::string DebugString() const {
    std::string res;
    // 3d-todo need to iterate over k (layer correctly.)
    for (int i = 7; i >= 0; --i) {
      for (int j = 0; j < 8; ++j) {
        if (get(i, j, 0))
          res += '#';
        else
          res += '.';
      }
      res += '\n';
    }
    return res;
  }

  // Applies a mask to the bitboard (intersects).
  BitBoard& operator&=(const BitBoard& a) {
    board_lower_ &= a.board_lower_;
    board_middle_ &= a.board_middle_;
    board_upper_ &= a.board_upper_;
  }

  friend void swap(BitBoard& a, BitBoard& b) {
    using std::swap;
    swap(a.board_lower_, b.board_lower_);
    swap(a.board_middle_, b.board_middle_);
    swap(a.board_upper_, b.board_upper_);
  }

  // Returns union (bitwise OR) of two boards.
  friend BitBoard operator|(const BitBoard& a, const BitBoard& b) {
    return {BitBoard( a.board_lower_ | b.board_lower_,
                  a.board_middle_ | b.board_middle_,
                  a.board_upper_ | b.board_upper_)};
  }

  // Returns intersection (bitwise AND) of two boards.
  friend BitBoard operator&(const BitBoard& a, const BitBoard& b) {
    return {BitBoard( a.board_lower_ & b.board_lower_,
                      a.board_middle_ & b.board_middle_,
                      a.board_upper_ & b.board_upper_)};

  }

  // Returns bitboard with one bit reset.
  friend BitBoard operator-(const BitBoard& a, const BoardSquare& b) {
    return {BitBoard( a.board_lower_ & ~b.as_board(),
                      a.board_middle_ & ~b.as_board(),
                      a.board_upper_ & ~b.as_board())};
  }

  // Returns difference (bitwise AND-NOT) of two boards.
  friend BitBoard operator-(const BitBoard& a, const BitBoard& b) {
    return {BitBoard( a.board_lower_ & ~b.board_lower_,
                      a.board_middle_ & ~b.board_middle_,
                      a.board_upper_ & ~b.board_upper_)};
  }

 private:
  std::uint64_t board_lower_ = 0;
  std::uint64_t board_middle_ = 0;
  std::uint64_t board_upper_ = 0;


};


// TODO 3d - masure updated
// Jesse TODO
class Move {
 public:
  enum class Promotion : std::uint8_t { None, Queen, Rook, Bishop, Knight };
  Move() = default;
  constexpr Move(BoardSquare from, BoardSquare to)
      : data_(to.as_int() + (from.as_int() << 6)) {}
  constexpr Move(BoardSquare from, BoardSquare to, Promotion promotion)
      : data_(to.as_int() + (from.as_int() << 6) +
              (static_cast<uint8_t>(promotion) << 12)) {}
  Move(const std::string& str, bool black = false);
  Move(const char* str, bool black = false) : Move(std::string(str), black) {}

  BoardSquare to() const { return BoardSquare(data_ & kToMask); }
  BoardSquare from() const { return BoardSquare((data_ & kFromMask) >> 6); }
  Promotion promotion() const { return Promotion((data_ & kPromoMask) >> 12); }

  void SetTo(BoardSquare to) { data_ = (data_ & ~kToMask) | to.as_int(); }
  void SetFrom(BoardSquare from) {
    data_ = (data_ & ~kFromMask) | (from.as_int() << 6);
  }
  void SetPromotion(Promotion promotion) {
    data_ = (data_ & ~kPromoMask) | (static_cast<uint8_t>(promotion) << 12);
  }
  // 0 .. 16384, knight promotion and no promotion is the same.
  uint16_t as_packed_int() const;

  // 0 .. 1857, to use in neural networks.
  // Transform is a bit field which describes a transform to be applied to the
  // the move before converting it to an index.
  uint16_t as_nn_index(int transform) const;

  explicit operator bool() const { return data_ != 0; }
  bool operator==(const Move& other) const { return data_ == other.data_; }

  void Mirror() { data_ ^= 0b111000111000; }

  std::string as_string() const {
    std::string res = from().as_string() + to().as_string();
    switch (promotion()) {
      case Promotion::None:
        return res;
      case Promotion::Queen:
        return res + 'q';
      case Promotion::Rook:
        return res + 'r';
      case Promotion::Bishop:
        return res + 'b';
      case Promotion::Knight:
        return res + 'n';
    }
    assert(false);
    return "Error!";
  }

 private:
  uint16_t data_ = 0;
  // Move, using the following encoding:
  // bits 0..5 "to"-square
  // bits 6..11 "from"-square
  // bits 12..14 promotion value


  enum Masks : uint16_t {
    kToMask =    0b0000000000111111,
    kFromMask =  0b0000111111000000,
    kPromoMask = 0b0111000000000000,
  };
};

using MoveList = std::vector<Move>;

}  // namespace lczero
