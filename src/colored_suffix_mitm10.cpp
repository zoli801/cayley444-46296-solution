#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Exact colour-state MITM for short suffixes.  A single radius-5 ball around
// the common solved state is reused for every puzzle.  For an incumbent suffix
// of length 11 or 12, a radius-4 or radius-5 forward ball respectively proves
// whether a replacement at least two moves shorter exists.

using Perm = std::array<uint8_t, 96>;
using State = std::array<uint8_t, 96>;

struct Row {
    int id = -1;
    std::vector<uint8_t> moves;
};

struct Word {
    Perm perm{};
    uint32_t packed = 0;
    uint8_t len = 0;
};

struct GoalRecord {
    uint64_t hash = 0;
    uint32_t word_index = 0;
};

struct Improvement {
    int suffix_length = 0;
    std::vector<uint8_t> bridge;
};

struct WindowSpec {
    int row = -1;
    int start = -1;
    int end = -1;
};

struct WindowBridge {
    int start = -1;
    int end = -1;
    std::vector<uint8_t> moves;
};

static std::array<Perm, 24> generators;
static std::array<std::array<uint64_t, 6>, 96> zobrist;
static std::vector<Row> rows;
static std::vector<State> initial_states;
static State central_state;

constexpr uint64_t BLOOM_BITS = 1ULL << 27; // 16 MiB, no false negatives.

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static int move_id(const std::string& token) {
    if (token.empty()) throw std::runtime_error("empty move token");
    const int neg = token[0] == '-' ? 1 : 0;
    if (int(token.size()) != neg + 2) throw std::runtime_error("bad move " + token);
    const int axis = token[neg] == 'f' ? 0 : token[neg] == 'r' ? 1 : token[neg] == 'd' ? 2 : -1;
    const int layer = token[neg + 1] - '0';
    if (axis < 0 || layer < 0 || layer > 3) throw std::runtime_error("bad move " + token);
    return axis * 8 + layer * 2 + neg;
}

static std::string move_name(int id) {
    static constexpr char axes[] = {'f', 'r', 'd'};
    std::string result;
    if (id & 1) result.push_back('-');
    result.push_back(axes[id / 8]);
    result.push_back(char('0' + (id % 8) / 2));
    return result;
}

static std::vector<uint8_t> parse_path(const std::string& text) {
    std::vector<uint8_t> result;
    if (text.empty()) return result;
    size_t begin = 0;
    while (true) {
        const size_t end = text.find('.', begin);
        result.push_back(uint8_t(move_id(text.substr(begin, end == std::string::npos ? end : end - begin))));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

static std::string format_path(const std::vector<uint8_t>& path) {
    std::ostringstream out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) out << '.';
        out << move_name(path[i]);
    }
    return out.str();
}

static void parse_number_array(const std::string& text, size_t pos, uint8_t* output, int upper) {
    for (int i = 0; i < 96; ++i) {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n')) ++pos;
        int value = 0;
        bool saw = false;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            saw = true;
            value = value * 10 + text[pos++] - '0';
        }
        if (!saw || value < 0 || value >= upper) throw std::runtime_error("bad numeric array");
        output[i] = uint8_t(value);
        if (i != 95) {
            if (pos >= text.size() || text[pos] != ',') throw std::runtime_error("bad numeric separator");
            ++pos;
        }
    }
}

static void load_puzzle_info(const std::string& path) {
    const std::string text = slurp(path);
    const std::string central_needle = "\"central_state\":[";
    const size_t cp = text.find(central_needle);
    if (cp == std::string::npos) throw std::runtime_error("missing central state");
    parse_number_array(text, cp + central_needle.size(), central_state.data(), 6);
    for (int id = 0; id < 24; ++id) {
        const std::string needle = "\"" + move_name(id) + "\":[";
        const size_t pos = text.find(needle);
        if (pos == std::string::npos) throw std::runtime_error("missing generator " + move_name(id));
        parse_number_array(text, pos + needle.size(), generators[id].data(), 96);
    }
    uint64_t seed = 0x6a09e667f3bcc909ULL;
    for (int i = 0; i < 96; ++i) {
        for (int c = 0; c < 6; ++c) {
            seed += 0x9e3779b97f4a7c15ULL;
            zobrist[i][c] = mix64(seed);
        }
    }
}

static void load_rows(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t comma = line.find(',');
        const int id = std::stoi(line.substr(0, comma));
        if (id != int(rows.size())) throw std::runtime_error("noncontiguous candidate IDs");
        rows.push_back(Row{id, parse_path(line.substr(comma + 1))});
    }
}

static State parse_color_state(const std::string& text) {
    State result{};
    size_t pos = 0;
    for (int i = 0; i < 96; ++i) {
        int value = 0;
        bool saw = false;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            saw = true;
            value = value * 10 + text[pos++] - '0';
        }
        if (!saw || value < 0 || value >= 6) throw std::runtime_error("bad color");
        result[i] = uint8_t(value);
        if (i != 95) {
            if (pos >= text.size() || text[pos] != ',') throw std::runtime_error("bad test state");
            ++pos;
        }
    }
    return result;
}

static void load_test(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t comma = line.find(',');
        const int id = std::stoi(line.substr(0, comma));
        const size_t q1 = line.find('"', comma + 1);
        const size_t q2 = line.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) throw std::runtime_error("bad test CSV");
        if (id != int(initial_states.size())) throw std::runtime_error("noncontiguous test IDs");
        initial_states.push_back(parse_color_state(line.substr(q1 + 1, q2 - q1 - 1)));
    }
}

static Perm identity_perm() {
    Perm p{};
    for (int i = 0; i < 96; ++i) p[i] = uint8_t(i);
    return p;
}

static Perm append_move(const Perm& current, int move) {
    Perm out{};
    const Perm& g = generators[move];
    for (int i = 0; i < 96; ++i) out[i] = current[g[i]];
    return out;
}

static State apply_move(const State& state, int move) {
    State out{};
    const Perm& g = generators[move];
    for (int i = 0; i < 96; ++i) out[i] = state[g[i]];
    return out;
}

static uint64_t apply_perm_hash(const State& state, const Perm& p, State& out) {
    uint64_t h = 0;
    for (int i = 0; i < 96; ++i) {
        out[i] = state[p[i]];
        h ^= zobrist[i][out[i]];
    }
    return h;
}

static void generate_words_dfs(std::vector<Word>& out, int max_depth, const Perm& current,
                               int depth, int last_axis, int last_layer,
                               int current_layer_count, int current_layer_sign,
                               uint32_t packed) {
    out.push_back(Word{current, packed, uint8_t(depth)});
    if (depth == max_depth) return;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != last_axis) {
            for (int layer = 0; layer < 4; ++layer) {
                for (int sign = 0; sign < 2; ++sign) {
                    const int id = axis * 8 + layer * 2 + sign;
                    generate_words_dfs(out, max_depth, append_move(current, id), depth + 1,
                                       axis, layer, 1, sign,
                                       packed | (uint32_t(id) << (5 * depth)));
                }
            }
        } else {
            if (current_layer_sign == 0 && current_layer_count == 1) {
                const int id = axis * 8 + last_layer * 2;
                generate_words_dfs(out, max_depth, append_move(current, id), depth + 1,
                                   axis, last_layer, 2, 0,
                                   packed | (uint32_t(id) << (5 * depth)));
            }
            for (int layer = last_layer + 1; layer < 4; ++layer) {
                for (int sign = 0; sign < 2; ++sign) {
                    const int id = axis * 8 + layer * 2 + sign;
                    generate_words_dfs(out, max_depth, append_move(current, id), depth + 1,
                                       axis, layer, 1, sign,
                                       packed | (uint32_t(id) << (5 * depth)));
                }
            }
        }
    }
}

static std::vector<Word> generate_words(int max_depth) {
    std::vector<Word> result;
    result.reserve(max_depth == 5 ? 3512239 : 182599);
    generate_words_dfs(result, max_depth, identity_perm(), 0, -1, -1, 0, 0, 0);
    return result;
}

static std::vector<uint8_t> unpack_word(const Word& word) {
    std::vector<uint8_t> out;
    for (int i = 0; i < word.len; ++i) out.push_back(uint8_t((word.packed >> (5 * i)) & 31));
    return out;
}

static std::vector<uint8_t> normalize_same_axis(const std::vector<uint8_t>& path) {
    std::vector<uint8_t> result;
    size_t i = 0;
    while (i < path.size()) {
        const int axis = path[i] / 8;
        int exp[4] = {0, 0, 0, 0};
        size_t j = i;
        while (j < path.size() && path[j] / 8 == axis) {
            const int layer = (path[j] % 8) / 2;
            exp[layer] = (exp[layer] + ((path[j] & 1) ? 3 : 1)) & 3;
            ++j;
        }
        for (int layer = 0; layer < 4; ++layer) {
            const int positive = axis * 8 + layer * 2;
            if (exp[layer] == 1) result.push_back(uint8_t(positive));
            else if (exp[layer] == 2) {
                result.push_back(uint8_t(positive));
                result.push_back(uint8_t(positive));
            } else if (exp[layer] == 3) {
                result.push_back(uint8_t(positive + 1));
            }
        }
        i = j;
    }
    return result;
}

static std::vector<uint8_t> construct_bridge(const Word& forward, const Word& from_goal) {
    std::vector<uint8_t> result = unpack_word(forward);
    const std::vector<uint8_t> backward = unpack_word(from_goal);
    for (auto it = backward.rbegin(); it != backward.rend(); ++it) result.push_back(uint8_t((*it) ^ 1));
    return normalize_same_axis(result);
}

static void bloom_add(std::vector<uint64_t>& bloom, uint64_t h) {
    const uint64_t a = h & (BLOOM_BITS - 1);
    const uint64_t b = mix64(h ^ 0x9e3779b97f4a7c15ULL) & (BLOOM_BITS - 1);
    bloom[a >> 6] |= 1ULL << (a & 63);
    bloom[b >> 6] |= 1ULL << (b & 63);
}

static bool bloom_maybe(const std::vector<uint64_t>& bloom, uint64_t h) {
    const uint64_t a = h & (BLOOM_BITS - 1);
    const uint64_t b = mix64(h ^ 0x9e3779b97f4a7c15ULL) & (BLOOM_BITS - 1);
    return ((bloom[a >> 6] >> (a & 63)) & 1U) && ((bloom[b >> 6] >> (b & 63)) & 1U);
}

static std::vector<State> path_states_for(int row_id) {
    const auto& path = rows[row_id].moves;
    std::vector<State> states(path.size() + 1);
    states[0] = initial_states[row_id];
    for (size_t i = 0; i < path.size(); ++i) states[i + 1] = apply_move(states[i], path[i]);
    if (states.back() != central_state) throw std::runtime_error("input candidate unsolved at row " + std::to_string(row_id));
    return states;
}

static Improvement search_suffix(const State& start, int suffix_length,
                                 const std::vector<Word>& words,
                                 const std::vector<GoalRecord>& goal_records,
                                 const std::vector<uint64_t>& bloom) {
    const int max_bridge = suffix_length - 2;
    const int max_forward = std::max(0, max_bridge - 5);
    Improvement best;
    best.suffix_length = suffix_length;
    best.bridge.resize(size_t(suffix_length), 255);
    State forward_state{};
    State goal_state{};
    for (uint32_t wi = 0; wi < words.size(); ++wi) {
        const Word& fw = words[wi];
        if (fw.len > max_forward) continue;
        const uint64_t h = apply_perm_hash(start, fw.perm, forward_state);
        if (!bloom_maybe(bloom, h)) continue;
        const auto first = std::lower_bound(goal_records.begin(), goal_records.end(), h,
            [](const GoalRecord& rec, uint64_t value) { return rec.hash < value; });
        for (auto it = first; it != goal_records.end() && it->hash == h; ++it) {
            const Word& goal_word = words[it->word_index];
            if (int(fw.len) + int(goal_word.len) > max_bridge) continue;
            apply_perm_hash(central_state, goal_word.perm, goal_state);
            if (goal_state != forward_state) continue;
            std::vector<uint8_t> bridge = construct_bridge(fw, goal_word);
            if (int(bridge.size()) > max_bridge) continue;
            State replay = start;
            for (uint8_t move : bridge) replay = apply_move(replay, move);
            if (replay != central_state) throw std::runtime_error("internal bridge replay failed");
            if (bridge.size() < best.bridge.size() ||
                (bridge.size() == best.bridge.size() && bridge < best.bridge)) {
                best.bridge = std::move(bridge);
            }
        }
    }
    if (best.bridge.size() >= size_t(suffix_length)) best.bridge.clear();
    return best;
}

static void build_target_ball(const State& target, const std::vector<Word>& words,
                              std::vector<GoalRecord>& records,
                              std::vector<uint64_t>& bloom) {
    records.clear();
    records.reserve(words.size());
    std::fill(bloom.begin(), bloom.end(), 0);
    State state{};
    for (uint32_t wi = 0; wi < words.size(); ++wi) {
        const uint64_t h = apply_perm_hash(target, words[wi].perm, state);
        records.push_back(GoalRecord{h, wi});
        bloom_add(bloom, h);
    }
    std::sort(records.begin(), records.end(), [](const GoalRecord& a, const GoalRecord& b) {
        return a.hash < b.hash;
    });
}

static std::vector<uint8_t> search_window(const State& start, const State& target,
                                          int window_length,
                                          const std::vector<Word>& words,
                                          const std::vector<GoalRecord>& target_records,
                                          const std::vector<uint64_t>& bloom) {
    const int max_bridge = window_length - 2;
    const int max_forward = std::max(0, max_bridge - 5);
    std::vector<uint8_t> best(size_t(window_length), 255);
    State forward_state{};
    State target_state{};
    for (uint32_t wi = 0; wi < words.size(); ++wi) {
        const Word& fw = words[wi];
        if (fw.len > max_forward) continue;
        const uint64_t h = apply_perm_hash(start, fw.perm, forward_state);
        if (!bloom_maybe(bloom, h)) continue;
        const auto first = std::lower_bound(target_records.begin(), target_records.end(), h,
            [](const GoalRecord& rec, uint64_t value) { return rec.hash < value; });
        for (auto it = first; it != target_records.end() && it->hash == h; ++it) {
            const Word& target_word = words[it->word_index];
            if (int(fw.len) + int(target_word.len) > max_bridge) continue;
            apply_perm_hash(target, target_word.perm, target_state);
            if (target_state != forward_state) continue;
            std::vector<uint8_t> bridge = construct_bridge(fw, target_word);
            if (int(bridge.size()) > max_bridge) continue;
            State replay = start;
            for (uint8_t move : bridge) replay = apply_move(replay, move);
            if (replay != target) throw std::runtime_error("internal window bridge replay failed");
            if (bridge.size() < best.size() ||
                (bridge.size() == best.size() && bridge < best)) {
                best = std::move(bridge);
            }
        }
    }
    if (best.size() >= size_t(window_length)) best.clear();
    return best;
}

static std::vector<WindowSpec> load_window_specs(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open window spec " + path);
    std::string line;
    std::getline(in, line);
    std::vector<WindowSpec> specs;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream fields(line);
        std::string row_text, start_text, end_text;
        if (!std::getline(fields, row_text, '\t') ||
            !std::getline(fields, start_text, '\t') ||
            !std::getline(fields, end_text, '\t')) {
            throw std::runtime_error("bad window spec line");
        }
        WindowSpec spec{std::stoi(row_text), std::stoi(start_text), std::stoi(end_text)};
        if (spec.row < 0 || spec.row >= int(rows.size()) || spec.start < 0 ||
            spec.end <= spec.start || spec.end > int(rows[spec.row].moves.size()) ||
            spec.end - spec.start > 12) {
            throw std::runtime_error("invalid window spec " + line);
        }
        specs.push_back(spec);
    }
    std::sort(specs.begin(), specs.end(), [](const WindowSpec& a, const WindowSpec& b) {
        if (a.row != b.row) return a.row < b.row;
        if (a.end != b.end) return a.end < b.end;
        return a.start < b.start;
    });
    specs.erase(std::unique(specs.begin(), specs.end(), [](const WindowSpec& a, const WindowSpec& b) {
        return a.row == b.row && a.start == b.start && a.end == b.end;
    }), specs.end());
    return specs;
}

static int run_window_specs(const std::string& spec_path,
                            const std::vector<Word>& words,
                            const std::string& output_path,
                            const std::string& relations_path,
                            const std::chrono::steady_clock::time_point started) {
    const std::vector<WindowSpec> specs = load_window_specs(spec_path);
    std::vector<std::vector<WindowBridge>> bridges(rows.size());
    std::vector<GoalRecord> target_records;
    std::vector<uint64_t> bloom(BLOOM_BITS / 64);
    size_t cursor = 0;
    int target_balls = 0;
    int improving_windows = 0;
    while (cursor < specs.size()) {
        const int row_id = specs[cursor].row;
        const int end = specs[cursor].end;
        size_t group_end = cursor + 1;
        while (group_end < specs.size() && specs[group_end].row == row_id && specs[group_end].end == end) {
            ++group_end;
        }
        const auto group_started = std::chrono::steady_clock::now();
        const std::vector<State> positions = path_states_for(row_id);
        build_target_ball(positions[end], words, target_records, bloom);
        ++target_balls;
        for (size_t index = cursor; index < group_end; ++index) {
            const WindowSpec& spec = specs[index];
            std::vector<uint8_t> bridge = search_window(
                positions[spec.start], positions[spec.end], spec.end - spec.start,
                words, target_records, bloom);
            if (!bridge.empty()) {
                bridges[row_id].push_back(WindowBridge{spec.start, spec.end, std::move(bridge)});
                ++improving_windows;
            }
        }
        const auto group_done = std::chrono::steady_clock::now();
        std::cout << "window_group=" << target_balls << " row=" << row_id << " end=" << end
                  << " starts=" << (group_end - cursor) << " improvements="
                  << bridges[row_id].size() << " seconds="
                  << std::chrono::duration<double>(group_done - group_started).count() << std::endl;
        cursor = group_end;
    }

    std::ofstream out(output_path);
    std::ofstream rel(relations_path);
    if (!out || !rel) throw std::runtime_error("cannot create window outputs");
    out << "initial_state_id,path\n";
    rel << "row\tstart\tend\told_len\tnew_len\tgain\told_path\tnew_path\n";
    long long before_total = 0;
    long long after_total = 0;
    int improved_rows = 0;
    int selected_relations = 0;
    for (const Row& row : rows) {
        const int n = int(row.moves.size());
        before_total += n;
        std::vector<int> dp(n + 1, 0);
        std::vector<int> chosen(n, -1);
        for (int position = n - 1; position >= 0; --position) {
            dp[position] = dp[position + 1];
            for (int index = 0; index < int(bridges[row.id].size()); ++index) {
                const WindowBridge& bridge = bridges[row.id][index];
                if (bridge.start != position) continue;
                const int gain = bridge.end - bridge.start - int(bridge.moves.size());
                if (gain + dp[bridge.end] > dp[position]) {
                    dp[position] = gain + dp[bridge.end];
                    chosen[position] = index;
                }
            }
        }
        std::vector<uint8_t> result;
        for (int position = 0; position < n;) {
            const int index = chosen[position];
            if (index < 0) {
                result.push_back(row.moves[position++]);
                continue;
            }
            const WindowBridge& bridge = bridges[row.id][index];
            std::vector<uint8_t> old_piece(row.moves.begin() + bridge.start, row.moves.begin() + bridge.end);
            rel << row.id << '\t' << bridge.start << '\t' << bridge.end << '\t'
                << old_piece.size() << '\t' << bridge.moves.size() << '\t'
                << (old_piece.size() - bridge.moves.size()) << '\t'
                << format_path(old_piece) << '\t' << format_path(bridge.moves) << '\n';
            result.insert(result.end(), bridge.moves.begin(), bridge.moves.end());
            position = bridge.end;
            ++selected_relations;
        }
        result = normalize_same_axis(result);
        State replay = initial_states[row.id];
        for (uint8_t move : result) replay = apply_move(replay, move);
        if (replay != central_state) throw std::runtime_error("window output unsolved row " + std::to_string(row.id));
        after_total += result.size();
        if (result.size() < row.moves.size()) ++improved_rows;
        out << row.id << ',' << format_path(result) << '\n';
    }
    const auto done = std::chrono::steady_clock::now();
    std::cout << "window_specs=" << specs.size() << '\n'
              << "target_balls=" << target_balls << '\n'
              << "improving_windows=" << improving_windows << '\n'
              << "selected_relations=" << selected_relations << '\n'
              << "improved_rows=" << improved_rows << '\n'
              << "before_total=" << before_total << '\n'
              << "after_total=" << after_total << '\n'
              << "total_gain=" << (before_total - after_total) << '\n'
              << "total_seconds=" << std::chrono::duration<double>(done - started).count() << '\n';
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 9 && argc != 10) {
        std::cerr << "usage: colored_suffix_mitm10 puzzle_info candidate test out.csv relations.tsv min_suffix max_suffix min_path_len [comma_ids|@window_specs.tsv]\n";
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    load_puzzle_info(argv[1]);
    load_rows(argv[2]);
    load_test(argv[3]);
    if (rows.size() != initial_states.size()) throw std::runtime_error("row count mismatch");
    const int min_suffix = std::stoi(argv[6]);
    const int max_suffix = std::stoi(argv[7]);
    const int min_path_len = std::stoi(argv[8]);
    if (min_suffix < 2 || max_suffix > 12 || min_suffix > max_suffix) {
        throw std::runtime_error("suffix range must satisfy 2 <= min <= max <= 12");
    }
    const bool window_mode = argc == 10 && argv[9][0] == '@';
    std::unordered_set<int> selected_ids;
    if (argc == 10 && !window_mode) {
        std::istringstream ids(argv[9]);
        std::string item;
        while (std::getline(ids, item, ',')) if (!item.empty()) selected_ids.insert(std::stoi(item));
    }

    std::vector<Word> words = generate_words(5);
    if (words.size() != 3512239) {
        throw std::runtime_error("unexpected radius-5 word count " + std::to_string(words.size()));
    }
    if (window_mode) {
        return run_window_specs(std::string(argv[9] + 1), words, argv[4], argv[5], started);
    }
    std::vector<GoalRecord> goal_records;
    goal_records.reserve(words.size());
    std::vector<uint64_t> bloom(BLOOM_BITS / 64);
    State state{};
    for (uint32_t wi = 0; wi < words.size(); ++wi) {
        const uint64_t h = apply_perm_hash(central_state, words[wi].perm, state);
        goal_records.push_back(GoalRecord{h, wi});
        bloom_add(bloom, h);
    }
    std::sort(goal_records.begin(), goal_records.end(), [](const GoalRecord& a, const GoalRecord& b) {
        return a.hash < b.hash;
    });
    const auto ball_ready = std::chrono::steady_clock::now();
    std::cout << "radius5_words=" << words.size() << " goal_ball_seconds="
              << std::chrono::duration<double>(ball_ready - started).count() << std::endl;

    std::vector<std::vector<uint8_t>> outputs(rows.size());
    std::vector<std::string> relation_lines(rows.size());
    long long before_total = 0;
    long long after_total = 0;
    int searched_rows = 0;
    int improved_rows = 0;

    for (const Row& row : rows) {
        before_total += row.moves.size();
        outputs[row.id] = row.moves;
        if (int(row.moves.size()) < min_path_len ||
            (!selected_ids.empty() && !selected_ids.count(row.id))) {
            after_total += outputs[row.id].size();
            continue;
        }
        const std::vector<State> positions = path_states_for(row.id);
        const auto row_started = std::chrono::steady_clock::now();
        ++searched_rows;
        int best_suffix = 0;
        std::vector<uint8_t> best_bridge;
        for (int suffix = min_suffix; suffix <= max_suffix; ++suffix) {
            if (suffix > int(row.moves.size())) continue;
            Improvement candidate = search_suffix(positions[row.moves.size() - suffix], suffix,
                                                  words, goal_records, bloom);
            if (candidate.bridge.empty()) continue;
            const int candidate_total = int(row.moves.size()) - suffix + int(candidate.bridge.size());
            const int best_total = best_suffix == 0
                ? int(row.moves.size())
                : int(row.moves.size()) - best_suffix + int(best_bridge.size());
            if (candidate_total < best_total ||
                (candidate_total == best_total && candidate.bridge < best_bridge)) {
                best_suffix = suffix;
                best_bridge = std::move(candidate.bridge);
            }
        }
        if (best_suffix != 0) {
            std::vector<uint8_t> old_suffix(row.moves.end() - best_suffix, row.moves.end());
            std::vector<uint8_t> result(row.moves.begin(), row.moves.end() - best_suffix);
            result.insert(result.end(), best_bridge.begin(), best_bridge.end());
            result = normalize_same_axis(result);
            State replay = initial_states[row.id];
            for (uint8_t move : result) replay = apply_move(replay, move);
            if (replay != central_state) throw std::runtime_error("rewritten row unsolved " + std::to_string(row.id));
            if (result.size() < outputs[row.id].size()) {
                std::ostringstream line;
                line << row.id << '\t' << best_suffix << '\t' << old_suffix.size() << '\t'
                     << best_bridge.size() << '\t' << (row.moves.size() - result.size()) << '\t'
                     << format_path(old_suffix) << '\t' << format_path(best_bridge);
                relation_lines[row.id] = line.str();
                outputs[row.id] = std::move(result);
                ++improved_rows;
            }
        }
        after_total += outputs[row.id].size();
        const auto row_done = std::chrono::steady_clock::now();
        std::cout << "row=" << row.id << " len=" << row.moves.size()
                  << " new_len=" << outputs[row.id].size() << " seconds="
                  << std::chrono::duration<double>(row_done - row_started).count() << std::endl;
    }

    std::ofstream out(argv[4]);
    std::ofstream rel(argv[5]);
    if (!out || !rel) throw std::runtime_error("cannot create outputs");
    out << "initial_state_id,path\n";
    rel << "row\tsuffix_len\told_len\tnew_len\tgain\told_path\tnew_path\n";
    for (const Row& row : rows) {
        out << row.id << ',' << format_path(outputs[row.id]) << '\n';
        if (!relation_lines[row.id].empty()) rel << relation_lines[row.id] << '\n';
    }
    const auto done = std::chrono::steady_clock::now();
    std::cout << "searched_rows=" << searched_rows << '\n'
              << "improved_rows=" << improved_rows << '\n'
              << "before_total=" << before_total << '\n'
              << "after_total=" << after_total << '\n'
              << "total_gain=" << (before_total - after_total) << '\n'
              << "total_seconds=" << std::chrono::duration<double>(done - started).count() << '\n';
    return 0;
}
