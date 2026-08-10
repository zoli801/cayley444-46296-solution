// Exact bounded state splicing for CayleyPy 4x4x4 submission paths.
//
// For each pair of checkpoints on an existing valid path, this program looks
// for a shorter path of at most --radius moves between the *actual sticker
// states*.  A meet-in-the-middle search (2+2 for radius 4, 2+3 for radius 5,
// or 3+3 for radius 6)
// makes this substantially cheaper than a separate BFS from every checkpoint.
// Equality uses all 96 stickers; hashes are only an acceleration mechanism.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int kWidth = 96;
using Perm = std::array<std::uint8_t, kWidth>;
using State = std::array<std::uint8_t, kWidth>;

struct ArrayHash {
  std::size_t operator()(const Perm& value) const noexcept {
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (int offset = 0; offset < kWidth; offset += 8) {
      std::uint64_t word = 0;
      for (int i = 0; i < 8; ++i) {
        word |= static_cast<std::uint64_t>(value[offset + i]) << (8 * i);
      }
      word ^= word >> 30;
      word *= 0xbf58476d1ce4e5b9ULL;
      word ^= word >> 27;
      word *= 0x94d049bb133111ebULL;
      hash ^= word ^ (word >> 31);
      hash = (hash << 17) | (hash >> 47);
      hash *= 0x9e3779b97f4a7c15ULL;
    }
    return static_cast<std::size_t>(hash ^ (hash >> 32));
  }
};

// Sticker colors are 0..5, so two stickers fit losslessly in one byte.  This
// halves hash-table key storage while preserving exact equality.
struct PackedState {
  std::array<std::uint64_t, 6> words{};
  bool operator==(const PackedState&) const = default;
};

struct PackedStateHash {
  std::size_t operator()(const PackedState& value) const noexcept {
    std::uint64_t hash = 0x243f6a8885a308d3ULL;
    for (std::uint64_t word : value.words) {
      word ^= word >> 30;
      word *= 0xbf58476d1ce4e5b9ULL;
      word ^= word >> 27;
      word *= 0x94d049bb133111ebULL;
      word ^= word >> 31;
      hash ^= word + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return static_cast<std::size_t>(hash ^ (hash >> 32));
  }
};

struct MoveSet {
  std::vector<std::string> names;
  std::vector<Perm> perms;
  std::unordered_map<std::string, int> indices;
};

struct BallEntry {
  Perm perm{};
  Perm inverse{};
  std::vector<std::uint8_t> word;
};

struct CsvTable {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
};

struct TestCase {
  std::string id;
  State initial{};
};

struct Solution {
  std::string id;
  std::vector<std::uint8_t> moves;
};

struct ForwardCandidate {
  int cost = std::numeric_limits<int>::max();
  int origin = -1;
  int ball_index = -1;
};

struct Previous {
  int origin = -1;
  std::vector<std::uint8_t> word;
};

struct PuzzleResult {
  std::vector<std::uint8_t> moves;
  int old_length = 0;
  int candidate_edges = 0;
};

std::string read_text(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (quoted) {
      if (ch == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          quoted = false;
        }
      } else {
        field.push_back(ch);
      }
    } else if (ch == ',') {
      fields.push_back(std::move(field));
      field.clear();
    } else if (ch == '"') {
      quoted = true;
    } else if (ch != '\r') {
      field.push_back(ch);
    }
  }
  if (quoted) throw std::runtime_error("unterminated CSV quote");
  fields.push_back(std::move(field));
  return fields;
}

CsvTable read_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path);
  CsvTable table;
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty CSV: " + path);
  table.header = parse_csv_line(line);
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    auto row = parse_csv_line(line);
    if (row.size() != table.header.size()) {
      throw std::runtime_error("CSV field count mismatch in " + path);
    }
    table.rows.push_back(std::move(row));
  }
  return table;
}

int column_index(const CsvTable& table, const std::string& name) {
  const auto found = std::find(table.header.begin(), table.header.end(), name);
  if (found == table.header.end()) throw std::runtime_error("missing CSV column " + name);
  return static_cast<int>(found - table.header.begin());
}

std::vector<int> parse_integer_list(const std::string& text) {
  std::vector<int> values;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) ||
                                 text[pos] == ',')) {
      ++pos;
    }
    if (pos == text.size()) break;
    bool negative = false;
    if (text[pos] == '-') {
      negative = true;
      ++pos;
    }
    if (pos == text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
      throw std::runtime_error("invalid integer list");
    }
    int value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
      value = value * 10 + (text[pos] - '0');
      ++pos;
    }
    values.push_back(negative ? -value : value);
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    if (pos < text.size() && text[pos] != ',') throw std::runtime_error("invalid list delimiter");
  }
  return values;
}

MoveSet load_moves(const std::string& path) {
  const std::string text = read_text(path);
  const std::size_t marker = text.find("\"generators\"");
  if (marker == std::string::npos) throw std::runtime_error("missing generators object");
  std::size_t pos = text.find('{', marker);
  if (pos == std::string::npos) throw std::runtime_error("malformed generators object");
  ++pos;

  MoveSet moves;
  while (true) {
    while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) ||
                                 text[pos] == ',')) {
      ++pos;
    }
    if (pos >= text.size() || text[pos] == '}') break;
    if (text[pos] != '"') throw std::runtime_error("malformed generator name");
    const std::size_t name_end = text.find('"', pos + 1);
    if (name_end == std::string::npos) throw std::runtime_error("malformed generator name");
    const std::string name = text.substr(pos + 1, name_end - pos - 1);
    const std::size_t array_start = text.find('[', name_end);
    const std::size_t array_end = text.find(']', array_start);
    if (array_start == std::string::npos || array_end == std::string::npos) {
      throw std::runtime_error("malformed generator permutation");
    }
    const auto values = parse_integer_list(text.substr(array_start + 1, array_end - array_start - 1));
    if (values.size() != kWidth) throw std::runtime_error("generator has wrong width: " + name);
    Perm perm{};
    std::array<bool, kWidth> used{};
    for (int i = 0; i < kWidth; ++i) {
      if (values[i] < 0 || values[i] >= kWidth || used[values[i]]) {
        throw std::runtime_error("generator is not a permutation: " + name);
      }
      perm[i] = static_cast<std::uint8_t>(values[i]);
      used[values[i]] = true;
    }
    moves.indices.emplace(name, static_cast<int>(moves.names.size()));
    moves.names.push_back(name);
    moves.perms.push_back(perm);
    pos = array_end + 1;
  }
  if (moves.names.empty()) throw std::runtime_error("no generators loaded");
  return moves;
}

Perm identity_perm() {
  Perm result{};
  for (int i = 0; i < kWidth; ++i) result[i] = static_cast<std::uint8_t>(i);
  return result;
}

Perm compose(const Perm& first, const Perm& second) {
  Perm result{};
  for (int i = 0; i < kWidth; ++i) result[i] = first[second[i]];
  return result;
}

Perm inverse_perm(const Perm& perm) {
  Perm inverse{};
  for (int i = 0; i < kWidth; ++i) inverse[perm[i]] = static_cast<std::uint8_t>(i);
  return inverse;
}

State apply_move(const State& state, const Perm& perm) {
  State result{};
  for (int i = 0; i < kWidth; ++i) result[i] = state[perm[i]];
  return result;
}

PackedState pack_applied(const State& state, const Perm& perm) {
  PackedState packed;
  for (int chunk = 0; chunk < 6; ++chunk) {
    std::uint64_t word = 0;
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      const int output = 16 * chunk + 2 * byte_index;
      const std::uint8_t low = state[perm[output]];
      const std::uint8_t high = state[perm[output + 1]];
      const std::uint8_t byte = static_cast<std::uint8_t>(low | (high << 4));
      word |= static_cast<std::uint64_t>(byte) << (8 * byte_index);
    }
    packed.words[chunk] = word;
  }
  return packed;
}

// A cheap lossy index for exact states.  Equal full states necessarily have
// equal signatures; unequal states that collide remain separate entries and
// are disambiguated by equal_applied below.  This reduces the common-case key
// construction from 96 sticker reads to 24 without sacrificing completeness.
std::uint64_t signature_applied(const State& state, const Perm& perm) {
  std::uint64_t signature = 0xcbf29ce484222325ULL;
  for (int face = 0; face < 6; ++face) {
    for (int local : {0, 1, 5, 6}) {
      signature ^= state[perm[16 * face + local]];
      signature *= 0x100000001b3ULL;
    }
  }
  return signature;
}

bool equal_applied(const State& first_state,
                   const Perm& first_perm,
                   const State& second_state,
                   const Perm& second_perm) {
  for (int output = 0; output < kWidth; ++output) {
    if (first_state[first_perm[output]] != second_state[second_perm[output]]) {
      return false;
    }
  }
  return true;
}

std::vector<BallEntry> build_ball(const MoveSet& moves, int radius) {
  BallEntry identity;
  identity.perm = identity_perm();
  identity.inverse = identity.perm;
  std::vector<BallEntry> entries{identity};
  std::vector<int> frontier{0};
  std::unordered_set<Perm, ArrayHash> seen;
  seen.reserve(200000);
  seen.insert(identity.perm);

  for (int depth = 1; depth <= radius; ++depth) {
    std::vector<int> next;
    for (int entry_index : frontier) {
      // Copy: entries.push_back below can reallocate the vector.
      const BallEntry parent = entries[entry_index];
      for (int move = 0; move < static_cast<int>(moves.perms.size()); ++move) {
        Perm perm = compose(parent.perm, moves.perms[move]);
        if (!seen.insert(perm).second) continue;
        BallEntry child;
        child.perm = perm;
        child.inverse = inverse_perm(perm);
        child.word = parent.word;
        child.word.push_back(static_cast<std::uint8_t>(move));
        entries.push_back(std::move(child));
        next.push_back(static_cast<int>(entries.size() - 1));
      }
    }
    frontier = std::move(next);
  }
  return entries;
}

State parse_state(const std::string& text) {
  const auto values = parse_integer_list(text);
  if (values.size() != kWidth) throw std::runtime_error("state has wrong width");
  State state{};
  for (int i = 0; i < kWidth; ++i) {
    if (values[i] < 0 || values[i] > 15) throw std::runtime_error("state value out of range");
    state[i] = static_cast<std::uint8_t>(values[i]);
  }
  return state;
}

std::vector<std::uint8_t> parse_path(const std::string& path, const MoveSet& moves) {
  std::vector<std::uint8_t> result;
  if (path.empty()) return result;
  std::size_t start = 0;
  while (start <= path.size()) {
    const std::size_t end = path.find('.', start);
    const std::string name = path.substr(start, end == std::string::npos ? end : end - start);
    const auto found = moves.indices.find(name);
    if (found == moves.indices.end()) throw std::runtime_error("unknown move in path: " + name);
    result.push_back(static_cast<std::uint8_t>(found->second));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return result;
}

std::string format_path(const std::vector<std::uint8_t>& path, const MoveSet& moves) {
  std::string result;
  for (std::size_t i = 0; i < path.size(); ++i) {
    if (i) result.push_back('.');
    result += moves.names[path[i]];
  }
  return result;
}

std::vector<TestCase> load_tests(const std::string& path) {
  const CsvTable table = read_csv(path);
  const int id_col = column_index(table, "initial_state_id");
  const int state_col = column_index(table, "initial_state");
  std::vector<TestCase> tests;
  tests.reserve(table.rows.size());
  for (const auto& row : table.rows) tests.push_back({row[id_col], parse_state(row[state_col])});
  return tests;
}

std::vector<Solution> load_solutions(const std::string& path, const MoveSet& moves) {
  const CsvTable table = read_csv(path);
  const int id_col = column_index(table, "initial_state_id");
  const int path_col = column_index(table, "path");
  std::vector<Solution> solutions;
  solutions.reserve(table.rows.size());
  for (const auto& row : table.rows) solutions.push_back({row[id_col], parse_path(row[path_col], moves)});
  return solutions;
}

std::vector<State> replay_checkpoints(const State& initial,
                                      const std::vector<std::uint8_t>& path,
                                      const MoveSet& moves) {
  std::vector<State> states;
  states.reserve(path.size() + 1);
  states.push_back(initial);
  for (std::uint8_t move : path) states.push_back(apply_move(states.back(), moves.perms[move]));
  return states;
}

PuzzleResult optimize_one(const State& initial,
                          const std::vector<std::uint8_t>& old_path,
                          const MoveSet& moves,
                          const std::vector<BallEntry>& forward_ball,
                          const std::vector<BallEntry>& backward_ball) {
  const auto checkpoints = replay_checkpoints(initial, old_path, moves);
  const int length = static_cast<int>(old_path.size());
  std::vector<int> distance(length + 1, std::numeric_limits<int>::max());
  std::vector<Previous> previous(length + 1);
  distance[0] = 0;

  std::unordered_map<PackedState, ForwardCandidate, PackedStateHash> forward;
  forward.reserve(static_cast<std::size_t>(length + 1) * forward_ball.size());
  int candidate_edges = 0;

  for (int endpoint = 1; endpoint <= length; ++endpoint) {
    const int origin = endpoint - 1;
    for (int ball_index = 0; ball_index < static_cast<int>(forward_ball.size()); ++ball_index) {
      const BallEntry& entry = forward_ball[ball_index];
      const int cost = distance[origin] + static_cast<int>(entry.word.size());
      const PackedState key = pack_applied(checkpoints[origin], entry.perm);
      auto [position, inserted] = forward.try_emplace(
          key, ForwardCandidate{cost, origin, ball_index});
      if (!inserted && cost < position->second.cost) {
        position->second = ForwardCandidate{cost, origin, ball_index};
      }
    }

    // The original edge always makes the dynamic program feasible.
    distance[endpoint] = distance[endpoint - 1] + 1;
    previous[endpoint] = Previous{endpoint - 1, {old_path[endpoint - 1]}};

    for (const BallEntry& suffix : backward_ball) {
      // If z followed by suffix equals checkpoint[endpoint], then
      // z = checkpoint[endpoint] acted on by inverse(suffix).
      const PackedState key = pack_applied(checkpoints[endpoint], suffix.inverse);
      const auto found = forward.find(key);
      if (found == forward.end()) continue;
      const ForwardCandidate& prefix = found->second;
      const BallEntry& first = forward_ball[prefix.ball_index];
      const int edge_length = static_cast<int>(first.word.size() + suffix.word.size());
      const int original_span = endpoint - prefix.origin;
      if (edge_length < original_span) ++candidate_edges;
      const int candidate = prefix.cost + static_cast<int>(suffix.word.size());
      if (candidate >= distance[endpoint]) continue;

      // Full-state keys are exact, but this explicit replay is a cheap local
      // invariant check and guards future changes to the packed representation.
      State reached = checkpoints[prefix.origin];
      for (std::uint8_t move : first.word) reached = apply_move(reached, moves.perms[move]);
      for (std::uint8_t move : suffix.word) reached = apply_move(reached, moves.perms[move]);
      if (reached != checkpoints[endpoint]) {
        throw std::runtime_error("internal splice equality check failed");
      }

      distance[endpoint] = candidate;
      previous[endpoint].origin = prefix.origin;
      previous[endpoint].word = first.word;
      previous[endpoint].word.insert(previous[endpoint].word.end(),
                                     suffix.word.begin(), suffix.word.end());
    }
  }

  std::vector<std::vector<std::uint8_t>> reversed_parts;
  for (int endpoint = length; endpoint > 0;) {
    const Previous& step = previous[endpoint];
    if (step.origin < 0 || step.origin >= endpoint) {
      throw std::runtime_error("invalid dynamic-program predecessor");
    }
    reversed_parts.push_back(step.word);
    endpoint = step.origin;
  }
  std::vector<std::uint8_t> result;
  for (auto part = reversed_parts.rbegin(); part != reversed_parts.rend(); ++part) {
    result.insert(result.end(), part->begin(), part->end());
  }
  if (static_cast<int>(result.size()) != distance[length]) {
    throw std::runtime_error("dynamic-program reconstruction mismatch");
  }
  return PuzzleResult{std::move(result), length, candidate_edges};
}

PuzzleResult optimize_one_signature(const State& initial,
                                    const std::vector<std::uint8_t>& old_path,
                                    const MoveSet& moves,
                                    const std::vector<BallEntry>& forward_ball,
                                    const std::vector<BallEntry>& backward_ball) {
  const auto checkpoints = replay_checkpoints(initial, old_path, moves);
  const int length = static_cast<int>(old_path.size());
  std::vector<int> distance(length + 1, std::numeric_limits<int>::max());
  std::vector<Previous> previous(length + 1);
  distance[0] = 0;

  // Multiple exact states can share a sampled signature.  We retain every
  // distinct full state in that bucket, merging only after a 96-sticker exact
  // comparison.  Hash collisions can therefore affect speed, never results.
  std::unordered_multimap<std::uint64_t, ForwardCandidate> forward;
  forward.reserve(static_cast<std::size_t>(length + 1) * forward_ball.size());
  int candidate_edges = 0;

  for (int endpoint = 1; endpoint <= length; ++endpoint) {
    const int origin = endpoint - 1;
    for (int ball_index = 0; ball_index < static_cast<int>(forward_ball.size()); ++ball_index) {
      const BallEntry& entry = forward_ball[ball_index];
      const int cost = distance[origin] + static_cast<int>(entry.word.size());
      const std::uint64_t key = signature_applied(checkpoints[origin], entry.perm);
      const auto range = forward.equal_range(key);
      bool merged = false;
      for (auto position = range.first; position != range.second; ++position) {
        ForwardCandidate& existing = position->second;
        const BallEntry& existing_entry = forward_ball[existing.ball_index];
        if (!equal_applied(checkpoints[origin], entry.perm,
                           checkpoints[existing.origin], existing_entry.perm)) {
          continue;
        }
        if (cost < existing.cost) {
          existing = ForwardCandidate{cost, origin, ball_index};
        }
        merged = true;
        break;
      }
      if (!merged) forward.emplace(key, ForwardCandidate{cost, origin, ball_index});
    }

    distance[endpoint] = distance[endpoint - 1] + 1;
    previous[endpoint] = Previous{endpoint - 1, {old_path[endpoint - 1]}};

    for (const BallEntry& suffix : backward_ball) {
      const std::uint64_t key = signature_applied(checkpoints[endpoint], suffix.inverse);
      const auto range = forward.equal_range(key);
      for (auto position = range.first; position != range.second; ++position) {
        const ForwardCandidate& prefix = position->second;
        const BallEntry& first = forward_ball[prefix.ball_index];
        if (!equal_applied(checkpoints[prefix.origin], first.perm,
                           checkpoints[endpoint], suffix.inverse)) {
          continue;
        }
        const int edge_length = static_cast<int>(first.word.size() + suffix.word.size());
        const int original_span = endpoint - prefix.origin;
        if (edge_length < original_span) ++candidate_edges;
        const int candidate = prefix.cost + static_cast<int>(suffix.word.size());
        if (candidate >= distance[endpoint]) continue;

        State reached = checkpoints[prefix.origin];
        for (std::uint8_t move : first.word) reached = apply_move(reached, moves.perms[move]);
        for (std::uint8_t move : suffix.word) reached = apply_move(reached, moves.perms[move]);
        if (reached != checkpoints[endpoint]) {
          throw std::runtime_error("internal signature splice equality check failed");
        }

        distance[endpoint] = candidate;
        previous[endpoint].origin = prefix.origin;
        previous[endpoint].word = first.word;
        previous[endpoint].word.insert(previous[endpoint].word.end(),
                                       suffix.word.begin(), suffix.word.end());
      }
    }
  }

  std::vector<std::vector<std::uint8_t>> reversed_parts;
  for (int endpoint = length; endpoint > 0;) {
    const Previous& step = previous[endpoint];
    if (step.origin < 0 || step.origin >= endpoint) {
      throw std::runtime_error("invalid signature DP predecessor");
    }
    reversed_parts.push_back(step.word);
    endpoint = step.origin;
  }
  std::vector<std::uint8_t> result;
  for (auto part = reversed_parts.rbegin(); part != reversed_parts.rend(); ++part) {
    result.insert(result.end(), part->begin(), part->end());
  }
  if (static_cast<int>(result.size()) != distance[length]) {
    throw std::runtime_error("signature DP reconstruction mismatch");
  }
  return PuzzleResult{std::move(result), length, candidate_edges};
}

void write_submission(const std::string& path,
                      const std::vector<Solution>& solutions,
                      const MoveSet& moves) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot create " + path);
  output << "initial_state_id,path\n";
  for (const Solution& solution : solutions) {
    output << solution.id << ',' << format_path(solution.moves, moves) << '\n';
  }
}

void usage(const char* program) {
  std::cerr << "usage: " << program
            << " --input submission.csv --output optimized.csv"
               " [--puzzle-info data/puzzle_info.json] [--test data/test.csv]"
               " [--radius 4|5|6|7] [--ids id1,id2,...]"
               " [--signature-index] [--quiet]\n";
}

}  // namespace

int main(int argc, char** argv) try {
  std::string puzzle_path = "data/puzzle_info.json";
  std::string test_path = "data/test.csv";
  std::string input_path;
  std::string output_path;
  std::unordered_set<std::string> selected_ids;
  int radius = 5;
  bool quiet_output = false;
  bool signature_index = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--quiet") {
      quiet_output = true;
      continue;
    }
    if (option == "--signature-index") {
      signature_index = true;
      continue;
    }
    if (i + 1 >= argc) {
      usage(argv[0]);
      return 2;
    }
    const std::string value = argv[++i];
    if (option == "--puzzle-info") puzzle_path = value;
    else if (option == "--test") test_path = value;
    else if (option == "--input") input_path = value;
    else if (option == "--output") output_path = value;
    else if (option == "--radius") radius = std::stoi(value);
    else if (option == "--ids") {
      std::size_t start = 0;
      while (start <= value.size()) {
        const std::size_t end = value.find(',', start);
        const std::string id = value.substr(
            start, end == std::string::npos ? end : end - start);
        if (id.empty()) throw std::runtime_error("empty ID in --ids");
        selected_ids.insert(id);
        if (end == std::string::npos) break;
        start = end + 1;
      }
    }
    else {
      usage(argv[0]);
      return 2;
    }
  }
  if (input_path.empty() || output_path.empty() ||
      (radius != 4 && radius != 5 && radius != 6 && radius != 7)) {
    usage(argv[0]);
    return 2;
  }

  const MoveSet moves = load_moves(puzzle_path);
  const auto tests = load_tests(test_path);
  auto solutions = load_solutions(input_path, moves);
  if (tests.size() != solutions.size()) throw std::runtime_error("test/submission row count mismatch");

  std::unordered_map<std::string, State> initial_by_id;
  initial_by_id.reserve(tests.size());
  for (const TestCase& test : tests) {
    if (!initial_by_id.emplace(test.id, test.initial).second) {
      throw std::runtime_error("duplicate test id " + test.id);
    }
  }

  const int forward_radius = radius / 2;
  const int backward_radius = radius - forward_radius;
  const auto forward_ball = build_ball(moves, forward_radius);
  const auto backward_ball = build_ball(moves, backward_radius);
  std::cerr << "moves=" << moves.names.size() << " forward_ball=" << forward_ball.size()
            << " backward_ball=" << backward_ball.size() << '\n';

  long long old_score = 0;
  long long new_score = 0;
  int improved = 0;
  long long candidate_edges = 0;
  for (std::size_t row = 0; row < solutions.size(); ++row) {
    Solution& solution = solutions[row];
    const auto initial = initial_by_id.find(solution.id);
    if (initial == initial_by_id.end()) throw std::runtime_error("unknown submission id " + solution.id);
    if (!selected_ids.empty() && !selected_ids.contains(solution.id)) {
      old_score += solution.moves.size();
      new_score += solution.moves.size();
      continue;
    }
    PuzzleResult result = signature_index
        ? optimize_one_signature(initial->second, solution.moves, moves,
                                 forward_ball, backward_ball)
        : optimize_one(initial->second, solution.moves, moves,
                       forward_ball, backward_ball);
    old_score += result.old_length;
    new_score += result.moves.size();
    candidate_edges += result.candidate_edges;
    if (result.moves.size() < solution.moves.size()) {
      ++improved;
      if (!quiet_output)
        std::cerr << "improved id=" << solution.id << " "
                  << solution.moves.size() << " -> " << result.moves.size()
                  << '\n';
    }
    solution.moves = std::move(result.moves);
    if (!quiet_output && (row + 1) % 100 == 0) {
      std::cerr << "processed=" << (row + 1) << '/' << solutions.size()
                << " saved=" << (old_score - new_score) << '\n';
    }
  }

  // Validate that every rewritten path reaches exactly the same endpoint as
  // its input path.  The Python competition validator can then independently
  // check that endpoint against central_state.
  const auto old_solutions = load_solutions(input_path, moves);
  for (std::size_t i = 0; i < solutions.size(); ++i) {
    const State& initial = initial_by_id.at(solutions[i].id);
    const State old_final = replay_checkpoints(initial, old_solutions[i].moves, moves).back();
    const State new_final = replay_checkpoints(initial, solutions[i].moves, moves).back();
    if (old_final != new_final) {
      throw std::runtime_error("final-state mismatch after rewrite for id " + solutions[i].id);
    }
  }

  write_submission(output_path, solutions, moves);
  std::cout << "puzzles=" << solutions.size() << " old_score=" << old_score
            << " new_score=" << new_score << " saved=" << (old_score - new_score)
            << " improved=" << improved << " candidate_edges=" << candidate_edges << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
